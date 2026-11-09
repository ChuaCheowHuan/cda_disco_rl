"""Disco103 in place of PPO's loss: the bridge, the plumbing, and a real run.

The rule, its weights and its settings come from the installed `disco_rl`;
nothing here copies them.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_CURR_KL_COEFF_KEY,
    LEARNER_RESULTS_KL_KEY,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchDiagGaussian,
    TorchMultiDistribution,
)
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from cda_discorl import policy_prediction as pp
from cda_discorl.disco103 import bridge, learner

from conftest import RealRun, assert_declared_on_the_pi_trunk

#: The action space's shape, with the Box order gymnasium sorts it into.
SPACE = gym.spaces.Dict(
    {
        "category": gym.spaces.Discrete(3),
        "size": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "price": gym.spaces.Discrete(4),
    }
)


class _Child:
    """A stand-in for one component's action distribution."""

    def __init__(self, kl_value):
        self.kl_value = kl_value
        self.asked = []

    def kl(self, other):
        self.asked.append(other)
        return torch.full((3,), self.kl_value)


def _rollout(a=4, t=6, n=2, seed=0):
    rng = np.random.default_rng(seed)
    y, bins = bridge.PREDICTION_SIZE, bridge.NUM_BINS
    out = {
        "logits": rng.normal(size=(t + 1, n, a)),
        "y": rng.normal(size=(t + 1, n, y)),
        "z": rng.normal(size=(t + 1, n, a, y)),
        "aux_pi": rng.normal(size=(t + 1, n, a, a)),
        "q": rng.normal(size=(t + 1, n, a, bins)),
    }
    return dict(
        actions=rng.integers(a, size=(t + 1, n)),
        rewards=rng.normal(size=(t, n)),
        is_terminal=np.zeros((t, n), bool),
        agent_out=out,
        behaviour_logits=out["logits"],
        target_out=out,
    )


class TestBridge:
    def test_the_settings_are_the_published_rules(self):
        from disco_rl import agent

        published = agent.get_settings_disco()
        assert bridge.PREDICTION_SIZE == published.update_rule.net.prediction_size
        assert bridge.NUM_BINS == published.update_rule.num_bins
        assert bridge.HYPER_PARAMS == dict(published.hyper_params)

    def test_the_published_weights_load_whole_from_the_installed_package(self):
        params = bridge.load_weights()
        assert len(params) == 21
        assert sum(len(v) for v in params.values()) == 42

    def test_targets_have_the_rules_shapes_and_carry_state(self):
        a, t, n = 4, 6, 2
        b = bridge.Bridge()
        targets, new_state = b.targets(b.new_state(), **_rollout(a, t, n))
        assert targets["pi"].shape == (t, n, a)
        assert targets["y"].shape == targets["z"].shape == (t, n, bridge.PREDICTION_SIZE)
        assert targets["q_probs"].shape == (t, n, bridge.NUM_BINS)
        assert np.allclose(targets["q_probs"].sum(-1), 1.0, atol=1e-5)
        assert all(np.isfinite(v).all() for v in targets.values())
        assert float(new_state["adv_ema_state"].decay_product) < 1.0

    def test_the_jitted_unroll_matches_the_eager_one(self):
        # The unroll is jitted (#29); this pins its result to the eager path.
        b = bridge.Bridge()
        state = b.new_state()
        roll = _rollout(a=4, t=8, n=2)
        jitted, _ = b.targets(state, **roll)

        as_j = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        eager, _ = b._unroll(
            state,
            jnp.asarray(roll["actions"], jnp.int32),
            as_j(roll["rewards"]),
            jnp.asarray(roll["is_terminal"], bool),
            jax.tree.map(as_j, roll["agent_out"]),
            {"logits": as_j(roll["behaviour_logits"])},
            jax.tree.map(as_j, roll["target_out"]),
        )
        for key in ("pi", "y", "z"):
            assert np.allclose(jitted[key], np.asarray(eager[key]), atol=1e-4), key
        # `q_probs` is a two-hot over 601 bins: a float difference far below the
        # bin width moves mass between neighbouring bins, so compare the value
        # the distribution implies (its mean bin) rather than per-bin mass.
        bins = np.arange(bridge.NUM_BINS)
        assert np.allclose(
            np.asarray(jitted["q_probs"]) @ bins,
            np.asarray(eager["q_probs"]) @ bins,
            atol=1e-2,
        )

    def test_the_new_ema_states_equal_the_rules_own_init_state(self):
        # The bridge builds them from `MovingAverage` and the public settings;
        # this fails if that drifts from the rule's own initialisation (#25).
        b = bridge.Bridge()
        state = b.new_state()
        for key, rule_ema in (("adv_ema_state", b.rule._adv_ema), ("td_ema_state", b.rule._td_ema)):
            want = rule_ema.init_state()
            got = state[key]
            assert jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)),
                    got,
                    want,
                )
            ), key

    def test_the_torch_value_loss_matches_the_rules(self):
        # CE against the bridge's two-hot target is the rule's own value loss.
        import jax.numpy as jnp
        from disco_rl import utils
        from disco_rl.value_fns import value_utils

        roll = _rollout()
        q = jnp.asarray(roll["agent_out"]["q"], jnp.float32)
        actions = jnp.asarray(roll["actions"], jnp.int32)
        td = jnp.asarray(np.random.default_rng(1).normal(size=roll["rewards"].shape), jnp.float32)
        q_a = utils.batch_lookup(q, actions)[:-1]
        want = value_utils.value_loss_from_td(
            q_a,
            td,
            nonlinear_transform=True,
            categorical_value=True,
            max_abs_value=bridge.MAX_ABS_VALUE,
        )
        probs = torch.as_tensor(np.asarray(bridge.value_target_probs(q, actions, td)))
        got = -(probs * torch.log_softmax(torch.as_tensor(np.asarray(q_a)), -1)).sum(-1)
        assert np.allclose(got.numpy(), np.asarray(want), atol=1e-4)


class TestTimeMajor:
    def test_round_trip_pads_with_the_last_row_and_zeroes_bootstrap_rows(self):
        flat = np.arange(7.0)  # episodes of 3 and 4 rows
        tm = learner._to_time_major(flat, [3, 4])
        assert tm[:, 0].tolist() == [0, 1, 2, 2]
        assert tm[:, 1].tolist() == [3, 4, 5, 6]
        back = learner._from_time_major(tm, [3, 4])
        assert back.tolist() == [0, 1, 0, 3, 4, 5, 0]


class TestPPOKlTerm:
    """The KL penalty `size` keeps from PPO (#10)."""

    def test_it_is_kl_of_prev_given_curr(self):
        curr, prev = _Child(0.0), _Child(0.5)
        kl = learner._non_discrete_kl({"size": (curr, prev)}, lambda x: x.mean())
        assert kl == pytest.approx(0.5)
        # PPO's direction is KL(prev || curr), as `prev.kl(curr)` in Ray.
        assert prev.asked == [curr]

    def test_it_is_zero_when_the_distributions_agree(self):
        pair = (_Child(0.0), _Child(0.0))
        kl = learner._non_discrete_kl({"size": pair}, lambda x: x.mean())
        assert float(kl) == 0.0


class TestRebuiltComponents:
    """PPO's part is rebuilt from the flat logits, not Ray's private struct (#24)."""

    def test_only_the_components_disco103_leaves_are_built(self):
        dists = learner._component_dists(SPACE, torch.zeros(3, 9), torch.zeros(3, 9), {"category"})
        assert set(dists) == {"size", "price"}
        curr, prev = dists["size"]
        assert isinstance(curr, TorchDiagGaussian) and isinstance(prev, TorchDiagGaussian)
        assert isinstance(dists["price"][0], TorchCategorical)

    def test_the_rebuilt_log_probs_match_the_multi_distribution(self):
        torch.manual_seed(0)
        flat = torch.randn(5, 9)
        actions = {
            "category": torch.tensor([0, 1, 2, 0, 1]),
            "price": torch.tensor([0, 3, 1, 2, 0]),
            "size": torch.randn(5, 1),
        }
        multi = TorchMultiDistribution(
            {
                "category": TorchCategorical.from_logits(flat[:, 0:3]),
                "price": TorchCategorical.from_logits(flat[:, 3:7]),
                "size": TorchDiagGaussian.from_logits(flat[:, 7:9]),
            }
        )
        rebuilt = {}
        for name, space, sl in pp.component_slices(SPACE):
            cls = TorchCategorical if isinstance(space, gym.spaces.Discrete) else TorchDiagGaussian
            rebuilt[name] = cls.from_logits(flat[:, sl]).logp(actions[name])
        assert torch.allclose(sum(rebuilt.values()), multi.logp(actions), atol=1e-6)
        assert rebuilt["size"].shape == (5,)


class TestInARealRun(RealRun):
    """policy_0 learns by Disco103; policy_1 stays plain PPO. Two epochs, so the targets are
    read by more than one pass over the batch."""

    settings = {"disco103": True}
    overrides = {"num_epochs": 2}

    def test_every_discrete_component_has_finite_disco_losses(self):
        metrics = self.result["learners"]["policy_0"]
        module = self.learner.module["policy_0"].unwrapped()
        for name, _ in module._d103_slices:
            for term in ("pi", "y", "z", "aux", "q"):
                assert np.isfinite(float(metrics[f"discorl_d103_{name}_{term}_loss"])), (name, term)
        assert np.isfinite(float(metrics["total_loss"]))

    def test_the_heads_are_declared_on_the_trunk_they_read(self):
        module = self.learner.module["policy_0"].unwrapped()
        assert_declared_on_the_pi_trunk(module, module.disco_heads)

    def test_the_critic_reports_its_explained_variance(self):
        # train.compare reads it; without it a Disco103 row has no critic column.
        assert np.isfinite(float(self.result["learners"]["policy_0"]["vf_explained_var"]))

    def test_the_continuous_component_keeps_ppo(self):
        assert "discorl_d103_size_policy_loss" in self.result["learners"]["policy_0"]

    def test_the_size_head_keeps_ppos_kl_penalty(self):
        # Without the KL term neither metric is logged, and Ray skips the
        # coefficient update because it cannot find the KL (#10).
        metrics = self.result["learners"]["policy_0"]
        assert np.isfinite(float(metrics[LEARNER_RESULTS_KL_KEY]))
        assert LEARNER_RESULTS_CURR_KL_COEFF_KEY in metrics

    def test_the_other_module_is_plain_ppo(self):
        assert not any("d103" in k for k in self.result["learners"]["policy_1"])
        assert not hasattr(self.learner.module["policy_1"].unwrapped(), "disco_heads")

    def test_the_meta_state_advanced_once_per_update(self):
        streams = self.learner._d103_streams["policy_0"]
        for state in streams.values():
            assert float(state["adv_ema_state"].decay_product) == pytest.approx(0.99)

    def test_the_checkpoint_restores_targets_and_meta_state(self):
        with self.restored() as got:
            for k, v in self.learner._d103_targets["policy_0"].state_dict().items():
                assert torch.equal(got._d103_targets["policy_0"].state_dict()[k].cpu(), v.cpu()), k
            want = self.learner._d103_streams["policy_0"]["category"]["rnn_state"]
            have = got._d103_streams["policy_0"]["category"]["rnn_state"]
            assert np.allclose(
                np.asarray(want["lstm"]["meta_rnn_state"].hidden),
                np.asarray(have["lstm"]["meta_rnn_state"].hidden),
            )


class TestWithAdvEma(RealRun):
    """Disco103 with the advantage EMA, the combination `validate` allows (#19).

    The EMA reads advantages and V(s), so it must run after the targets are
    written; its placement is what this run checks.
    """

    settings = {"disco103": True, "adv_ema": True}

    def test_the_targets_are_written_before_the_advantage_ema(self):
        names = [type(c).__name__ for c in self.learner._learner_connector.connectors]
        assert names.index("Disco103Targets") < names.index("EmaAdvantages")

    def test_the_run_trains_with_finite_losses(self):
        metrics = self.result["learners"]["policy_0"]
        assert np.isfinite(float(metrics["total_loss"]))
        module = self.learner.module["policy_0"].unwrapped()
        for name, _ in module._d103_slices:
            assert np.isfinite(float(metrics[f"discorl_d103_{name}_pi_loss"]))


class TestWithARemoteRunner(RealRun):
    """Disco103 with a remote env runner, so its batches come from env-runner chunks (#16).

    A remote runner samples with its own timing and can split an episode across
    batches; the episode lengths the connector sees must still sum to the rows,
    or the rule would get misaligned trajectories.
    """

    settings = {"disco103": True}
    overrides = {"num_env_runners": 1}
    batches = []

    @classmethod
    def train_once(cls):
        original = learner.Disco103Learner._d103_write_targets

        def spy(self, module, module_id, mbatch, lens, reward_scale):
            if module_id == "policy_0":
                cls.batches.append((list(lens), int(mbatch[Columns.REWARDS].shape[0])))
            return original(self, module, module_id, mbatch, lens, reward_scale)

        learner.Disco103Learner._d103_write_targets = spy
        try:
            return cls.algo.train()
        finally:
            learner.Disco103Learner._d103_write_targets = original

    def test_each_batch_holds_exactly_the_rows_its_episodes_cover(self):
        assert self.batches
        for lens, rows in self.batches:
            assert sum(lens) == rows, (lens, rows)

    def test_the_run_trains_with_finite_disco_losses(self):
        metrics = self.result["learners"]["policy_0"]
        assert np.isfinite(float(metrics["total_loss"]))
        module = self.learner.module["policy_0"].unwrapped()
        for name, _ in module._d103_slices:
            assert np.isfinite(float(metrics[f"discorl_d103_{name}_pi_loss"]))
