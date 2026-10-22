"""What this package takes from `disco_rl`, imported in one place.

The policy is to import whatever `disco_rl` provides rather than reimplement
it. What stays local is what must be torch, because gradients flow through it
inside RLlib's Learner: the loss terms, the extra heads, and the in-place
update of a torch target network.

From `disco_rl` come the rule's published settings and loss costs
(`agent.get_settings_disco`), the moving-average decay its update rule uses
(`DiscoUpdateRule`), and the debiased moving average itself
(`utils.MovingAverage`). `disco103/bridge.py` imports the rule and its
weights on top of these.
"""

from __future__ import annotations

import inspect
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
import torch
from disco_rl import agent as disco_agent
from disco_rl import types
from disco_rl.update_rules import disco
from disco_rl.utils import MovingAverage

#: Disco103's published settings: its update rule, agent and loss costs.
SETTINGS = disco_agent.get_settings_disco()
HYPER_PARAMS: Dict[str, float] = dict(SETTINGS.hyper_params)

#: The decay of the advantage and TD moving averages in DiscoRL's update rule.
MOVING_AVERAGE_DECAY: float = (
    inspect.signature(disco.DiscoUpdateRule.__init__).parameters["moving_average_decay"].default
)


class EmaMoments:
    """One stream's debiased moving average of a value's first two moments.

    The arithmetic is `disco_rl.utils.MovingAverage` (debiased as Adam debiases
    its moments, with its own epsilons); this class only carries its state as
    plain floats, so it checkpoints as numbers, and converts torch tensors in
    and out. No gradient passes through it. MovingAverage computes in JAX's
    default float32.
    """

    def __init__(self, decay: float):
        self._ema = MovingAverage(jnp.zeros(()), decay=decay)
        self.moment1 = 0.0
        self.moment2 = 0.0
        self.decay_product = 1.0

    def _state(self) -> types.EmaState:
        return types.EmaState(
            moment1=jnp.float32(self.moment1),
            moment2=jnp.float32(self.moment2),
            decay_product=jnp.float32(self.decay_product),
        )

    @property
    def updated(self) -> bool:
        return self.decay_product != 1.0

    def update(self, values: torch.Tensor) -> None:
        new = self._ema.update_state(_to_jax(values), self._state(), None)
        self.moment1 = float(new.moment1)
        self.moment2 = float(new.moment2)
        self.decay_product = float(new.decay_product)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        out = self._ema.normalize(_to_jax(values), self._state())
        return torch.tensor(np.asarray(out), device=values.device, dtype=values.dtype)

    def mean_std(self) -> Tuple[float, float]:
        """The mean and the scale `normalize` divides by, read back from it.

        `normalize(x) = (x - mean) / scale`, so `scale = 1 / normalize(1)`
        without the mean, and `mean = -scale * normalize(0)`."""
        state = self._state()
        inv_scale = float(self._ema.normalize(jnp.float32(1.0), state, subtract_mean=False))
        scale = 1.0 / inv_scale
        return -scale * float(self._ema.normalize(jnp.float32(0.0), state)), scale

    def get_state(self) -> Dict[str, float]:
        return {
            "moment1": self.moment1,
            "moment2": self.moment2,
            "decay_product": self.decay_product,
        }

    def set_state(self, state: Dict[str, float]) -> None:
        self.moment1 = float(state["moment1"])
        self.moment2 = float(state["moment2"])
        self.decay_product = float(state["decay_product"])


def _to_jax(values: torch.Tensor):
    return jnp.asarray(values.detach().cpu().numpy(), jnp.float32)


def ema_update_(target: torch.nn.Module, online: torch.nn.Module, coeff: float) -> None:
    """`target = coeff * target + (1 - coeff) * online`, DiscoRL's target-parameter
    update, in place on torch parameters (so it cannot be the upstream JAX tree map)."""
    with torch.no_grad():
        for t, p in zip(target.parameters(), online.parameters()):
            t.mul_(coeff).add_(p.detach(), alpha=1.0 - coeff)
