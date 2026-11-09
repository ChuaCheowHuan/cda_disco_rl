"""The target policy component, piece by piece. Its real run is in `test_real_run.py`."""

import pytest
import torch
from torch import nn

from cda_discorl import target_policy as tp


class _Wrapped:
    def __init__(self, m):
        self.m = m

    def unwrapped(self):
        return self.m


class _Base:
    def apply_gradients(self, gradients_dict):
        self.applied = True


class _Learner(tp.TargetPolicyLearner, _Base):
    def __init__(self, online, target, coeff):
        self.module = {"p": _Wrapped(online)}
        self._targets = {"p": target}
        self._target_coeff = coeff


class TestEmaUpdate:
    def test_each_optimiser_step_moves_the_target_by_one_minus_coeff(self):
        online, target = nn.Linear(3, 2), nn.Linear(3, 2)
        before = [p.detach().clone() for p in target.parameters()]
        learner = _Learner(online, target, coeff=0.9)
        learner.apply_gradients({})
        assert learner.applied
        for t, b, o in zip(target.parameters(), before, online.parameters()):
            assert torch.allclose(t, 0.9 * b + 0.1 * o)

    def test_coeff_zero_copies_the_online_weights(self):
        online, target = nn.Linear(2, 2), nn.Linear(2, 2)
        _Learner(online, target, coeff=0.0).apply_gradients({})
        assert all(torch.equal(t, o) for t, o in zip(target.parameters(), online.parameters()))

    def test_nothing_happens_when_off(self):
        learner = _Learner(nn.Linear(3, 2), nn.Linear(3, 2), coeff=0.9)
        learner._targets = None
        learner.apply_gradients({})
        assert learner.applied


class _Named:
    """A module whose single parameter's name is chosen, to break the pairing."""

    def __init__(self, name):
        self.module = nn.Module()
        setattr(self.module, name, nn.Parameter(torch.zeros(2, 2)))

    def named_parameters(self):
        return self.module.named_parameters()

    def parameters(self):
        return self.module.parameters()


class TestMismatchedCopies:
    """A target that stopped matching the online module must fail loudly (#14)."""

    def test_a_different_parameter_count_is_refused(self):
        online = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
        with pytest.raises(ValueError, match="no longer matches"):
            _Learner(online, nn.Linear(3, 2), coeff=0.9).apply_gradients({})

    def test_different_parameter_names_are_refused(self):
        with pytest.raises(ValueError, match="no longer matches"):
            _Learner(_Named("w"), _Named("v"), coeff=0.9).apply_gradients({})
