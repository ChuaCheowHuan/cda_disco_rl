"""The policy prediction component, piece by piece. Its real run is in `test_real_run.py`."""

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.columns import Columns

from cda_discorl import policy_prediction as pp

SPACE = gym.spaces.Dict(
    {
        "category": gym.spaces.Discrete(3),
        "size": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "price": gym.spaces.Discrete(4),
    }
)


class _FakeModule:
    """A module stand-in: a known flat `ACTION_DIST_INPUTS` over `SPACE`."""

    def __init__(self):
        self._pp_slices = pp.discrete_slices(SPACE)
        self.encoder = None
        self.calls = []

    def _forward(self, batch):
        self.calls.append(batch)
        rows = batch[Columns.OBS].shape[0]
        return {Columns.ACTION_DIST_INPUTS: torch.arange(9.0).expand(rows, 9)}


class _Wrapped:
    def __init__(self, module):
        self._module = module

    def unwrapped(self):
        return self._module


class TestPieces:
    def test_discrete_slices_skip_the_box(self):
        # gymnasium sorts a Dict's keys: category, price, size.
        assert pp.discrete_slices(SPACE) == [("category", slice(0, 3)), ("price", slice(3, 7))]

    def test_gather_takes_the_discrete_logits_in_order(self):
        space = gym.spaces.Dict(
            {
                "a": gym.spaces.Discrete(2),
                "b": gym.spaces.Box(0, 1, (1,)),
                "c": gym.spaces.Discrete(3),
            }
        )
        logits = torch.arange(7.0).expand(2, 7)
        got = pp.gather_discrete(logits, pp.discrete_slices(space))
        assert got[0].tolist() == [0, 1, 4, 5, 6]

    def test_kl_is_zero_for_equal_logits_and_sums_over_components(self):
        t = torch.randn(4, 7)
        assert torch.allclose(pp.per_head_kl(t, t, [3, 4]), torch.zeros(4), atol=1e-6)
        p = torch.randn(4, 7)
        a = torch.distributions.Categorical(logits=t[:, :3])
        b = torch.distributions.Categorical(logits=p[:, :3])
        c = torch.distributions.Categorical(logits=t[:, 3:])
        d = torch.distributions.Categorical(logits=p[:, 3:])
        expected = torch.distributions.kl_divergence(a, b) + torch.distributions.kl_divergence(c, d)
        assert torch.allclose(pp.per_head_kl(t, p, [3, 4]), expected, atol=1e-6)

    def test_a_masked_action_in_the_target_contributes_nothing(self):
        t = torch.tensor([[0.0, -1.0e9, 0.0]])
        p = torch.tensor([[0.0, 5.0, 0.0]])
        assert torch.isfinite(pp.per_head_kl(t, p, [3])).all()


class TestTargets:
    """The next-step logits are a batch column, written once (#18, #31)."""

    def test_it_writes_the_discrete_logits_of_the_next_observation(self):
        module = _FakeModule()
        batch = {"p": {Columns.NEXT_OBS: torch.zeros(2, 5)}}
        out = pp.PolicyPredictionTargets(["p"])(
            rl_module={"p": _Wrapped(module)}, batch=batch, episodes=None
        )
        got = out["p"][pp.NEXT_LOGITS]
        # category 0:3 and price 3:7; the Box's 7:9 is not a discrete target.
        assert got.shape == (2, 7)
        assert got[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert len(module.calls) == 1

    def test_a_module_without_the_column_is_skipped(self):
        batch = {}
        out = pp.PolicyPredictionTargets(["p"])(
            rl_module={"p": _Wrapped(_FakeModule())}, batch=batch, episodes=None
        )
        assert out == {}
