"""The JAX side of Disco103, imported from the installed `disco_rl` package.

Everything that defines the rule comes from `disco_rl`: the meta-network and
its update (`DiscoUpdateRule`), its settings and loss costs
(`agent.get_settings_disco`), the published weights
(`update_rules/weights/disco_103.npz`) and the value transforms the rule's
value loss uses (`value_fns.value_utils`). This module only converts arrays
between numpy and JAX and keeps each stream's meta-state.

`Bridge.targets` takes one action component's time-major rollout as numpy
arrays and returns the targets the torch loss is trained toward. Each
(module, action component) pair is one stream with its own meta-RNN state and
advantage/TD moving averages, carried across updates as the rule expects.

The target network's outputs are computed in torch and handed in; the rule's
own target-parameter update is fed an empty tree and its result discarded,
since the torch Learner keeps that copy.

`heads` and `learner` import this module only when `disco103` is on, so a run
without it never builds the meta-network.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import rlax
from disco_rl import types, utils
from disco_rl.update_rules import disco
from disco_rl.value_fns import value_utils

from cda_discorl.upstream import HYPER_PARAMS, SETTINGS as _SETTINGS

#: What the torch side needs from the rule's published settings.
PREDICTION_SIZE = int(_SETTINGS.update_rule.net.prediction_size)
NUM_BINS = int(_SETTINGS.update_rule.num_bins)
MAX_ABS_VALUE = float(_SETTINGS.update_rule.max_abs_value)
HEAD_INIT_STD = float(_SETTINGS.net_settings.net_args.head_w_init_std)


def load_weights() -> Dict[str, Dict[str, np.ndarray]]:
    """The Disco103 meta-network parameters, from the installed package.

    The file is flat (`<module>/w`, `<module>/b`); Haiku wants them nested by
    module, as the upstream colabs unflatten them."""
    path = resources.files("disco_rl.update_rules") / "weights" / "disco_103.npz"
    with path.open("rb") as f:
        flat = dict(np.load(f))
    params: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in flat.items():
        module, name = key.rsplit("/", 1)
        params.setdefault(module, {})[name] = value
    return params


def value_target_probs(q_logits, actions, q_td):
    """The two-hot value target the rule's value loss trains `q[a]` toward.

    `value_utils.value_loss_from_td` with the rule's settings: the taken
    action's categorical value, plus the TD, through the signed hyperbolic
    transform, as a two-hot over the rule's support. `q_logits` is
    `[T + 1, B, A, bins]`, `actions` `[T + 1, B]`, `q_td` `[T, B]`."""
    q_a = utils.batch_lookup(q_logits, actions)[:-1]
    values = value_utils.get_values_from_net_outs(
        q_a, categorical_value=True, max_abs_value=MAX_ABS_VALUE, nonlinear_transform=True
    )
    down, _ = rlax.SIGNED_HYPERBOLIC_PAIR
    return value_utils.scalar_to_categorical_value(
        q_a.shape[-1], down(values + q_td), MAX_ABS_VALUE
    )


class Bridge:
    """Disco103 with its published weights, ready to produce targets."""

    def __init__(self):
        self.rule = disco.DiscoUpdateRule(**dict(_SETTINGS.update_rule))
        self.hyper_params = dict(HYPER_PARAMS)
        self.params = jax.tree.map(jnp.asarray, load_weights())

    def new_state(self) -> Dict[str, Any]:
        _, rnn_state = self.rule.init_params(jax.random.PRNGKey(0))
        return dict(
            rnn_state=rnn_state,
            adv_ema_state=self.rule._adv_ema.init_state(),
            td_ema_state=self.rule._td_ema.init_state(),
            target_params={},
        )

    def _unroll(self, meta_state, actions, rewards, is_terminal, agent_out, behaviour, target):
        # Not jitted: under `jit` a tracer fails the upstream meta-net's
        # `isinstance(logits, chex.Array)` check with current JAX.
        rollout = types.UpdateRuleInputs(
            observations=jnp.zeros(actions.shape),
            actions=actions,
            rewards=rewards,
            is_terminal=is_terminal,
            agent_out=agent_out,
            behaviour_agent_out=behaviour,
        )
        out, new_state = self.rule.unroll_meta_net(
            self.params,
            {},
            None,
            meta_state,
            rollout,
            self.hyper_params,
            lambda *_: (target, None),
            jax.random.PRNGKey(0),
        )
        keep = {k: out[k] for k in ("pi", "y", "z")}
        keep["q_probs"] = value_target_probs(agent_out["q"], actions, out["q_td"])
        return keep, new_state

    def targets(
        self, state, *, actions, rewards, is_terminal, agent_out, behaviour_logits, target_out
    ):
        """Targets for one stream. Arrays are time-major, `[T + 1, B, ...]` except
        `rewards` and `is_terminal`, `[T, B]`. Returns `(targets, new_state)`."""
        as_j = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        out, new_state = self._unroll(
            state,
            jnp.asarray(actions, jnp.int32),
            as_j(rewards),
            jnp.asarray(is_terminal, bool),
            jax.tree.map(as_j, agent_out),
            {"logits": as_j(behaviour_logits)},
            jax.tree.map(as_j, target_out),
        )
        new_state = dict(new_state, target_params={})
        return {k: np.asarray(v) for k, v in out.items()}, new_state


def state_to_numpy(state):
    return jax.tree.map(np.asarray, state)


def state_from_numpy(state):
    return jax.tree.map(jnp.asarray, state)
