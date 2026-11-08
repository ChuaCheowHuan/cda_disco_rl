"""The advantage EMA component, piece by piece. Its real run is in `test_real_run.py`."""

import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from cda_discorl import adv_ema, upstream


def _reference(batches, decay):
    """DiscoRL's MovingAverage written out on float64, to check the wrapper feeds it
    the right state. It computes in float32, hence the tolerance below."""
    m1 = m2 = 0.0
    prod = 1.0
    out = []
    for b in batches:
        m1 = decay * m1 + (1 - decay) * b.mean().item()
        m2 = decay * m2 + (1 - decay) * (b**2).mean().item()
        prod *= decay
        mean = m1 / (1 - prod)
        var = max(0.0, m2 / (1 - prod) - mean**2)
        out.append((b - mean) / ((var + 1e-12) ** 0.5 + 1e-6))
    return out


class TestMoments:
    def test_matches_discorls_moving_average(self):
        torch.manual_seed(0)
        batches = [torch.randn(32, dtype=torch.float64) * s + s for s in (1.0, 3.0, 0.1, 10.0)]
        m = upstream.EmaMoments(decay=0.9)
        got = []
        for b in batches:
            m.update(b)
            got.append(m.normalize(b))
        for a, b in zip(got, _reference(batches, 0.9)):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_the_first_batch_is_standardised_as_before(self):
        """Debiasing makes step one's moments the batch's own."""
        b = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
        m = upstream.EmaMoments(decay=0.99)
        m.update(b)
        expected = (b - b.mean()) / b.std(unbiased=False)
        assert torch.allclose(m.normalize(b), expected, atol=1e-5)

    def test_a_small_batch_keeps_its_scale_against_the_past(self):
        m = upstream.EmaMoments(decay=0.99)
        m.update(torch.randn(256) * 10.0)
        small = torch.randn(256) * 0.1
        m.update(small)
        assert m.normalize(small).std() < 0.1


class _Module:
    def __init__(self, values):
        self.values = values

    def unwrapped(self):
        return self

    def compute_values(self, batch):
        return self.values


class TestConnector:
    def test_normalises_only_the_selected_modules(self):
        targets = torch.tensor([1.0, 2.0, 4.0, 8.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        untouched = torch.tensor([9.0, 9.0, 9.0, 9.0])
        batch = {
            "policy_0": {
                Postprocessing.VALUE_TARGETS: targets,
                Postprocessing.ADVANTAGES: untouched,
            },
            "policy_1": {
                Postprocessing.VALUE_TARGETS: targets,
                Postprocessing.ADVANTAGES: untouched,
            },
        }
        rl_module = {"policy_0": _Module(values), "policy_1": _Module(values)}
        piece = adv_ema.EmaAdvantages(["policy_0"], decay=0.9)
        piece(rl_module=rl_module, batch=batch)

        expected = upstream.EmaMoments(0.9)
        expected.update(targets - values)
        assert torch.allclose(
            batch["policy_0"][Postprocessing.ADVANTAGES], expected.normalize(targets - values)
        )
        assert batch["policy_1"][Postprocessing.ADVANTAGES] is untouched

    def test_padding_is_left_out_of_the_moments(self):
        targets = torch.tensor([1.0, 3.0, 100.0])
        batch = {
            "policy_0": {
                Postprocessing.VALUE_TARGETS: targets,
                Columns.LOSS_MASK: torch.tensor([True, True, False]),
            }
        }
        piece = adv_ema.EmaAdvantages(["policy_0"], decay=0.9)
        piece(rl_module={"policy_0": _Module(torch.zeros(3))}, batch=batch)
        assert piece.moments["policy_0"].moment1 == pytest.approx(0.1 * 2.0)
