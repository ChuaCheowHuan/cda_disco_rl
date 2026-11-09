"""What a Continual Backprop reset does to a target copy (#27).

`target_policy` and `disco103` each keep an EMA copy of the module. CBP resets a
low-utility unit's weights in place, and the copy keeps the old unit until it
blends toward the new one at `1 - target_coeff` per optimiser step. This
measures that: force one replacement, then record the online-copy gap and the
target-policy KL for the optimiser steps that follow. The numbers feed the
decision in #17; the module skips when cda-cbp is not installed.
"""

import pytest

pytest.importorskip("cda_cbp")

import torch  # noqa: E402

from cda_cbp import learner as cbp_learner  # noqa: E402

from cda_discorl import target_policy as tp  # noqa: E402

from conftest import RealRun  # noqa: E402

#: Replace one unit per layer per optimiser step, from the first step on.
FORCE_ONE = {
    "replacement_rate": 1.0,
    "maturity_threshold": 0,
    "count_rule": "nature",
}

#: The measurement window, in optimiser steps (#27).
STEPS = 20


class _CbpReplacement(RealRun):
    """One forced CBP replacement, then `STEPS` optimiser steps of measurement."""

    extensions = ("discorl", "cbp")
    extra_settings = {"cbp": FORCE_ONE}
    #: Ten epochs over the tiny batch give more than `STEPS` optimiser steps in
    #: one iteration, with the meta-network targets computed only once.
    overrides = {"num_epochs": 10}

    records = []
    gaps = []
    kls = []

    #: The attribute holding `{module_id: copy}` on the Learner.
    copies_attr = ""

    @classmethod
    def copies(cls):
        return getattr(cls.learner, cls.copies_attr)["policy_0"]

    @classmethod
    def kl(cls):
        """The target-policy KL for the step just taken, or nan without it."""
        try:
            value = cls.learner.metrics.peek(("policy_0", tp.TARGET_KL_KEY))
            return float(value.detach() if hasattr(value, "detach") else value)
        except Exception:  # noqa: BLE001 - the metric is absent for disco103
            return float("nan")

    @classmethod
    def gap(cls):
        """Mean |online - copy| over the replaced unit's incoming weights."""
        record = cls.records[0]
        copy_weight = cls.copies().state_dict()[record["name"]]
        online = record["incoming"].weight.detach()
        idx = record["indices"]
        return float((online[idx] - copy_weight[idx]).abs().mean())

    @classmethod
    def train_once(cls):
        cls.records, cls.gaps, cls.kls = [], [], []
        original_replace = cbp_learner.select_and_replace
        original_apply = cls.learner.apply_gradients

        def replace(state, layer, corrected_utility, config, generator=None, optimizer=None):
            indices = original_replace(
                state,
                layer,
                corrected_utility,
                config,
                generator=generator,
                optimizer=optimizer,
            )
            if indices and not cls.records:
                name = f"{layer.name}.weight"
                cls.records.append(
                    {
                        "name": name,
                        "indices": list(indices),
                        "incoming": layer.incoming,
                        "online_after": layer.incoming.weight.detach().clone(),
                        "copy_before": cls.copies().state_dict()[name].detach().clone(),
                    }
                )
            return indices

        def apply(gradients_dict):
            original_apply(gradients_dict)
            if cls.records:
                cls.gaps.append(cls.gap())
                cls.kls.append(cls.kl())

        cbp_learner.select_and_replace = replace
        cls.learner.apply_gradients = apply
        try:
            return cls.algo.train()
        finally:
            cbp_learner.select_and_replace = original_replace
            del cls.learner.apply_gradients

    def test_a_unit_was_replaced(self):
        assert self.records, "CBP replaced nothing; the forced settings did not fire"
        assert self.records[0]["indices"]

    def test_the_copy_kept_the_old_unit_at_first(self):
        record = self.records[0]
        idx = record["indices"]
        gap = (record["online_after"][idx] - record["copy_before"][idx]).abs().mean()
        assert float(gap) > 0

    def test_it_catches_up_over_the_next_steps(self):
        assert len(self.gaps) >= STEPS, len(self.gaps)
        assert self.gaps[0] > 0
        # The copy blends at 1 - target_coeff per step, so the gap must shrink.
        assert self.gaps[-1] < self.gaps[0]


class TestStaleTargetPolicy(_CbpReplacement):
    """`target_policy` with CBP: the KL the stale copy contributes (#27)."""

    settings = {"target_policy": True}
    copies_attr = "_targets"

    def test_the_target_kl_is_recorded_after_the_replacement(self):
        kls = [k for k in self.kls if k == k]  # drop nan
        assert len(kls) >= STEPS
        assert all(k >= 0 for k in kls)


class TestStaleDisco103Targets(_CbpReplacement):
    """`disco103` with CBP: the rule's targets come from the stale copy (#27)."""

    settings = {"disco103": True}
    copies_attr = "_d103_targets"

    def test_the_rule_kept_training_after_the_replacement(self):
        assert torch.isfinite(torch.tensor(self.gaps)).all()
