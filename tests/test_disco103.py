"""Disco103 in place of PPO's loss: the bridge, the plumbing, and a real run.

The rule, its weights and its settings come from the installed `disco_rl`;
nothing here copies them.
"""

import numpy as np
import pytest
import torch

from cda_discorl.disco103 import bridge, learner

from conftest import RealRun, assert_declared_on_the_pi_trunk


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
