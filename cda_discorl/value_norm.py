"""Value normalisation: the critic learns returns on a unit scale.

This market's value targets are tiny. Their standard deviation is about 0.004 in a
512-step episode, because rewards are fractions of starting capital. A critic
whose loss assumes a fixed scale, as DiscoRL's categorical critic and Disco103
do, is badly placed to see a signal that small; neither learned here. DiscoRL
itself never relies on raw scale; it normalises TD errors and advantages by
running moments (`disco_rl/value_fns`), with `disco_rl.utils.MovingAverage`.

With `value_norm` on, the value head predicts the *normalised* return,
`(G - mean) / std`, where mean and std come from a debiased EMA of the value
targets: `disco_rl`'s MovingAverage, through the same `upstream.EmaMoments`
the advantage EMA uses. The critic loss sees
normalised targets against that prediction. `compute_values` maps it back,
`mean + std * prediction`, so GAE, the advantages and every value RLlib reads
stay in reward units.

The moments are module buffers, so they are saved and restored with the
module's weights. They are not PopArt: when the moments move, the
denormalised value moves with them for one update. With a slow EMA that step is
small.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train.extensions import settings_for

from cda_discorl.adv_ema import EmaAdvantages
from cda_discorl.upstream import EmaMoments

NAME = "discorl"

#: Metrics logged per module: the moments in force for the update.
MEAN_KEY = "discorl_value_norm_mean"
STD_KEY = "discorl_value_norm_std"

#: EmaMoments' state, one float64 buffer each.
_MOMENTS = ("moment1", "moment2", "decay_product")


class ValueNormModule:
    """Module mixin: the value head's output is a normalised return.

    Inert unless `value_norm` is on. It must sit before any mixin that
    replaces the value head, so that its `compute_values` wraps theirs.
    """

    _vn_on = False
    #: While true, `compute_values` returns the head's normalised prediction.
    _vn_raw = False

    def setup(self):
        super().setup()
        s = type(self).extension_settings[NAME]
        if not s["value_norm"]:
            return
        self._vn_on = True
        self._vn_decay = float(s["value_norm_decay"])
        # Moments of the value targets, as EmaMoments keeps them, as buffers so
        # they travel with the weights.
        for name, start in zip(_MOMENTS, (0.0, 0.0, 1.0)):
            self.register_buffer(f"_vn_{name}", torch.tensor(start, dtype=torch.float64))

    def _vn_moments(self) -> EmaMoments:
        m = EmaMoments(self._vn_decay)
        m.set_state({n: float(getattr(self, f"_vn_{n}")) for n in _MOMENTS})
        return m

    def value_norm_stats(self):
        """`(mean, std)` of the value targets so far; `(0, 1)` before any update."""
        m = self._vn_moments()
        return m.mean_std() if m.updated else (0.0, 1.0)

    def update_value_norm(self, targets: torch.Tensor) -> None:
        m = self._vn_moments()
        m.update(targets)
        for name, value in m.get_state().items():
            getattr(self, f"_vn_{name}").fill_(value)

    def set_state(self, state) -> None:
        # RLlib's set_state casts float64 arrays to float32. Restore the moments
        # exactly as saved, so a resumed run continues from the same numbers.
        super().set_state(state)
        if not self._vn_on:
            return
        for name in _MOMENTS:
            key = f"_vn_{name}"
            if key in state:
                value = torch.as_tensor(np.asarray(state[key]), dtype=torch.float64)
                getattr(self, key).copy_(value)

    def compute_values(self, batch, embeddings=None):
        values = super().compute_values(batch, embeddings=embeddings)
        if not self._vn_on or self._vn_raw:
            return values
        mean, std = self.value_norm_stats()
        return mean + std * values


class ValueNormMoments(ConnectorV2):
    """After GAE (and after the advantage EMA when it is on), folds each selected
    module's value targets into its moments."""

    def __init__(self, module_ids: Iterable[str], **kwargs):
        super().__init__(**kwargs)
        self.module_ids = list(module_ids)

    def __call__(self, *, rl_module, batch, episodes=None, **kwargs):
        for module_id in self.module_ids:
            if module_id not in batch:
                continue
            mbatch = batch[module_id]
            targets = mbatch[Postprocessing.VALUE_TARGETS]
            if Columns.LOSS_MASK in mbatch:
                targets = targets[mbatch[Columns.LOSS_MASK]]
            rl_module[module_id].unwrapped().update_value_norm(targets)
        return batch


class ValueNormLearner:
    """Learner mixin: the critic loss runs on normalised targets and predictions.

    It must sit before every other mixin that reads `VALUE_TARGETS` in its
    loss, so they all see the normalised batch.
    """

    def build(self) -> None:
        super().build()
        s = settings_for(self.config.learner_config_dict, NAME)
        if not s.get("value_norm"):
            return
        # After the advantage EMA when it is on: it recomputes V(s) through
        # `compute_values`, which must still use the stats GAE bootstrapped with.
        anchor = EmaAdvantages if s.get("adv_ema") else GeneralAdvantageEstimation
        self._learner_connector.insert_after(anchor, ValueNormMoments(s["modules"]))

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        if not getattr(module, "_vn_on", False):
            return super()._policy_value_loss(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )
        mean, std = module.value_norm_stats()
        targets = batch[Postprocessing.VALUE_TARGETS]
        batch = {**batch, Postprocessing.VALUE_TARGETS: (targets - mean) / std}
        module._vn_raw = True
        try:
            loss = super()._policy_value_loss(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )
        finally:
            module._vn_raw = False
        self.metrics.log_value((module_id, MEAN_KEY), mean, window=1)
        self.metrics.log_value((module_id, STD_KEY), std, window=1)
        return loss
