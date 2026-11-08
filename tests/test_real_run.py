"""Real gym-CDA runs: the extension enabled with everything off, then with every component
that combines on at once.

The second run is where each component meets a real Learner, a real connector pipeline and
a remote env runner, so its checks are per component. `disco103` replaces PPO's loss and
excludes three of the others; its own real run is in `test_disco103.py`.
"""

import numpy as np
import pytest
import torch

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.checkpoint import _read_league_state, save_checkpoint
from gym_continuousDoubleAuction.train.model.moe_learner import (
    CDAPPOTorchLearner,
    CDAPPOTorchRLModule,
)

import cda_discorl
from cda_discorl import adv_ema, config
from cda_discorl import policy_prediction as pp
from cda_discorl import target_policy as tp
from cda_discorl import value_norm as vn

from conftest import RealRun, assert_declared_on_the_pi_trunk


class TestEnabledButInert(RealRun):
    """Enabled with no settings: plain PPO, and the checkpoint names the extension."""

    def test_the_learner_and_modules_are_gym_cdas_own(self):
        assert isinstance(self.learner, CDAPPOTorchLearner)
        assert type(self.algo.env_runner.module["policy_0"]) is CDAPPOTorchRLModule

    def test_both_modules_train_as_plain_ppo(self):
        for module_id in ("policy_0", "policy_1"):
            metrics = self.result["learners"][module_id]
            assert "total_loss" in metrics
            assert not any(k.startswith("discorl") for k in metrics)

    def test_the_checkpoint_records_the_extension(self):
        path = save_checkpoint(self.algo, self.cfg, 1)
        recorded = _read_league_state(path)[extensions.EXTENSIONS_STATE_KEY]
        assert recorded == {"discorl": cda_discorl.__version__}


#: Every component but disco103, on policy_0 only, with decays that show each update.
ALL_BUT_DISCO103 = {
    **{name: True for name in config.COMPONENTS if name != "disco103"},
    "target_coeff": 0.5,
    "adv_ema_decay": 0.9,
    "value_norm_decay": 0.9,
}


class TestEveryComponentButDisco103(RealRun):
    """policy_0 has every component but disco103; policy_1 stays plain PPO.

    Two iterations, so the moments move twice and the value-norm stats change between
    them. One remote env runner, which builds inference-only modules and receives weights
    from the Learner: an extra head missing from the inference-only list fails here.
    """

    settings = ALL_BUT_DISCO103
    overrides = {"num_env_runners": 1}
    iterations = 2

    @classmethod
    def before_training(cls):
        cls.module = cls.learner.module["policy_0"].unwrapped()
        cls.initial_target = {
            k: v.detach().clone() for k, v in cls.learner._targets["policy_0"].state_dict().items()
        }
        cls.stats_before, cls.stats_seen_by_adv_ema = [], []

    @classmethod
    def train_once(cls):
        # Records the value-norm stats `EmaAdvantages` sees, to compare with the ones GAE used.
        cls.stats_before.append(cls.module.value_norm_stats())
        original = adv_ema.EmaAdvantages.__call__

        def spy(self, **kwargs):
            cls.stats_seen_by_adv_ema.append(cls.module.value_norm_stats())
            return original(self, **kwargs)

        adv_ema.EmaAdvantages.__call__ = spy
        try:
            return cls.algo.train()
        finally:
            adv_ema.EmaAdvantages.__call__ = original

    # -- the run as a whole ----------------------------------------------------

    def test_both_iterations_train_with_finite_losses(self):
        for result in self.results:
            metrics = result["learners"]["policy_0"]
            assert np.isfinite(float(metrics["total_loss"]))
            for key in (tp.TARGET_KL_KEY, vn.STD_KEY, pp.POLICY_PREDICTION_LOSS_KEY):
                assert np.isfinite(float(metrics[key])), key
            assert float(metrics[tp.TARGET_KL_KEY]) >= 0
            assert float(metrics[pp.POLICY_PREDICTION_LOSS_KEY]) >= 0

    def test_the_other_module_is_plain_ppo(self):
        assert not any(k.startswith("discorl") for k in self.result["learners"]["policy_1"])
        other = self.learner.module["policy_1"].unwrapped()
        assert not hasattr(other, "aux_pi")
        assert not getattr(other, "_vn_on", False)
        assert list(self.learner._targets) == ["policy_0"]
        assert list(self.learner._adv_ema.moments) == ["policy_0"]

    def test_the_connector_pieces_are_where_they_must_be(self):
        names = [type(c).__name__ for c in self.learner._learner_connector.connectors]
        gae = names.index("GeneralAdvantageEstimation")
        # The value-norm moments move after the advantage EMA has read V(s) (components.md §3).
        assert names[gae + 1 : gae + 3] == ["EmaAdvantages", "ValueNormMoments"]
        assert names.count("AddNextObservationsFromEpisodesToTrainBatch") == 1
        assert names.index("AddNextObservationsFromEpisodesToTrainBatch") == (
            names.index("AddOneTsToEpisodesAndTruncate") + 1
        )

    # -- per component ---------------------------------------------------------

    def test_the_target_moved_and_lags_the_online_weights(self):
        target = self.learner._targets["policy_0"].state_dict()
        online = self.module.state_dict()
        moved = [
            k for k in self.initial_target if not torch.equal(target[k], self.initial_target[k])
        ]
        assert moved
        assert any(not torch.equal(target[k], online[k]) for k in moved)

    def test_the_advantage_and_value_moments_moved_once_per_iteration(self):
        assert self.learner._adv_ema.moments["policy_0"].decay_product == pytest.approx(0.81)
        assert float(self.module._vn_decay_product) == pytest.approx(0.81)

    def test_the_advantage_ema_sees_the_stats_gae_used(self):
        assert self.stats_seen_by_adv_ema == self.stats_before
        assert self.stats_before[1] != (0.0, 1.0), "the moments moved between the iterations"

    def test_the_prediction_head_is_declared_on_the_trunk_it_reads(self):
        assert_declared_on_the_pi_trunk(self.module, self.module.aux_pi)

    def test_the_checkpoint_restores_every_components_state(self):
        with self.restored() as got:
            for k, v in self.learner._targets["policy_0"].state_dict().items():
                assert torch.equal(got._targets["policy_0"].state_dict()[k].cpu(), v.cpu()), k
            assert (
                got._adv_ema.moments["policy_0"].get_state()
                == self.learner._adv_ema.moments["policy_0"].get_state()
            )
            module = got.module["policy_0"].unwrapped()
            # Exact: the moments are restored as float64, not RLlib's float32.
            assert module._vn_moment2.dtype == torch.float64
            assert module.value_norm_stats() == pytest.approx(
                self.module.value_norm_stats(), rel=1e-12
            )
            assert torch.equal(
                module.aux_pi.weight.detach().cpu(), self.module.aux_pi.weight.detach().cpu()
            )
