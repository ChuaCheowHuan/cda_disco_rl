# Components

What each component does, what it logs and what it saves. Every component is off by default and
applies only to the trainable modules listed in `modules`; [configuration.md](configuration.md)
has every setting and the rules between them, and [architecture.md](architecture.md) §3 the
Learner and module hooks each one uses.

The [README](../README.md) summarises them and their status, which comes from [findings.md](findings.md).

## 1. Target policy

DiscoRL's agent keeps target weights, an exponential moving average of its own. With
`target_policy` on, each selected module gets such a copy on the Learner, and the loss gains
`target_kl_cost * KL(pi_target || pi)`. After every optimiser step the copy moves
`1 - target_coeff` of the way to the online weights.

PPO's clipping limits how far one update moves the policy. This limits how far a run of updates
drifts from where the policy has recently been, which matters when the league keeps changing the
opponents. The KL is logged as `discorl_target_kl`, and the copy is saved with the Learner's state.

## 2. Advantage EMA

RLlib standardises each batch's advantages by that batch's own mean and standard deviation.
With `adv_ema` on, the selected modules use DiscoRL's normaliser instead: an exponential moving
average (decay `adv_ema_decay`) of the advantage's first and second moments over the whole run,
debiased as Adam debiases its moments. The moving average is `disco_rl.utils.MovingAverage`,
imported. The first batch is standardised exactly as before. Later
batches keep their scale relative to the past rather than being stretched to unit variance.

It runs right after RLlib's GAE step on the Learner. GAE writes standardised advantages and
drops the raw ones, so `EmaAdvantages` recomputes them as `value_targets - V(s)` with the same
weights, folds the batch into the debiased moments and normalises with them. Both steps are
`disco_rl.utils.MovingAverage`'s `update_state` and `normalize`, through `upstream.EmaMoments`,
which keeps the state as three plain floats. The moments are saved with the Learner's state, so a
resume continues them.

## 3. Value normalisation

With `value_norm` on, the value head predicts the return normalised by a debiased EMA (decay
`value_norm_decay`) of the value targets' mean and standard deviation. PPO's critic loss then runs
on normalised targets against that prediction. The moments come from the same imported
`MovingAverage`. `compute_values` maps the prediction back to
reward units, so GAE, the advantages and every metric RLlib reports are unchanged in meaning.

The moments are module buffers, saved and restored with the weights, and logged as
`discorl_value_norm_mean` and `discorl_value_norm_std`. They are not PopArt: when the moments move,
the value moves with them for one update.

`ValueNormMoments` updates the module's moments from the masked `VALUE_TARGETS` right after GAE,
or right after `EmaAdvantages` when `adv_ema` is on too. That order matters: `EmaAdvantages`
recomputes `V(s)` through `compute_values`, and it must use the stats GAE bootstrapped with, not
ones already updated from this batch.
In the loss, `ValueNormLearner` replaces the targets with `(T - mean) / std` and sets the
module's `_vn_raw` flag around the `super()` call, so PPO's critic loss compares normalised
targets with the head's normalised output. Everywhere else, `compute_values` returns
`mean + std * head`, so GAE and RLlib's metrics keep reward units.

The `(mean, std)` pair is cached on the module and refreshed only when the moments move: by
`ValueNormMoments`, and by `set_state` or `load_state_dict` on a restore. `compute_values` runs on
every GAE pass, every minibatch's loss and every `EmaAdvantages` pass, so recomputing it in JAX
each time was pure overhead (#30).

## 4. Policy prediction

DiscoRL's agent has an auxiliary output that predicts, for each action it could take now, the
policy it will have at the next step, trained toward the actual next-step policy. Nothing acts on
it; it shapes the shared representation.

This market's action is a Dict. So with `policy_prediction` on, the head is conditioned on the
`category` action and predicts the logits of every discrete component (category, price, price
offset, order slot) at the next step. The loss is the sum over those components of
KL(next-step policy || prediction), with terminal steps masked out. It is weighted by
`policy_prediction_cost` and logged as `discorl_policy_prediction_loss`.

It adds the next observation to the Learner's train batch when the run does not already carry it
(JEPA runs do). `PolicyPredictionTargets` then computes the next step's logits once per update
batch, right after GAE, and writes them into the batch; every minibatch of every epoch reads them
per row. They are therefore a pre-update target, as in DiscoRL, which is what makes several epochs
cheap (#31). It needs a stateless encoder, so it refuses `encoder_type: lstm`. The head lives
only on the Learner's copy of a module; env runners do not build it.

## 5. Disco103

Disco103 is the update rule DeepMind meta-learned and published with `disco_rl`. With `disco103`
on, the selected modules learn by it instead of PPO's loss. Everything that defines the rule comes
from the installed `disco_rl`: the meta-network (`DiscoUpdateRule`), its settings and loss costs
(`get_settings_disco`), the weights (`disco_103.npz`) and the value transforms. This package adds
only what a torch Learner needs around it:

- **Heads.** The rule reads more agent outputs than a policy: `y`, `z`, `aux_pi` and a
  categorical `q`. Each discrete action component (category, price, price offset, order slot)
  gets its own set, sized by the rule's settings, from the actor embedding. They exist only on the
  Learner's copy of a module.
- **Targets.** Once per update batch, right after GAE, the batch is put in time order and handed
  to the rule as numpy arrays. The rule returns its policy, `y`, `z` and value targets, which go
  back into the batch.
- **Losses.** In torch, against those targets, with the rule's costs: KL to each target, the
  policy-prediction KL, and cross-entropy for the value. These must be torch, because the
  gradients flow through the torch module.
- **The rest.** The continuous `size` component keeps PPO's clipped surrogate, its entropy bonus
  and, when `use_kl_loss` is on (Ray's default), its KL penalty; the scalar critic is PPO's too,
  since the rule handles one discrete action. Rewards are multiplied by
  `disco103_reward_scale` (default 100) first, because the rule was discovered on Atari-scale
  rewards. An EMA target network (the rule's `target_params_coeff`) and each stream's meta-RNN
  state are saved with the Learner.

It cannot be combined with `target_policy` or `policy_prediction`, which the rule supplies
itself, or with `value_norm`, which would change the critic it keeps for `size`. It needs a
stateless encoder. Its losses are logged per discrete component as
`discorl_d103_<component>_{pi,y,z,aux,q}_loss`, and the size head's as
`discorl_d103_size_policy_loss`. Each module grows to about 9.3M parameters against 0.26M, a run takes about
4× as long, and each run holds about 4 GB of memory.

The rule's meta-network unroll is jitted, one compilation per rollout length (#29). A 512 × 4
rollout takes about 0.04 s against 0.28 s eagerly, after about 1.2 s of compilation. The fork is
untouched: a small shim answers the one `isinstance(logits, chex.Array)` guard for the tracer
`jax.jit` passes.

The `size` loss rebuilds each component's distribution from its own slice of `ACTION_DIST_INPUTS`,
rather than reading Ray's private `TorchMultiDistribution._original_struct` (#24). The batch rows
are checked against the episode lengths before the rule sees them, so a misaligned batch fails
loudly instead of training on shifted trajectories (#16).

**The horizon.** The rule keeps its published `value_discount` 0.997 (about a 330-step horizon)
and `value_fn_td_lambda` 0.95 for the discrete actions' TD(λ) value targets, while GAE and the
scalar critic use the run's `gamma` 0.999 (about a 1,000-step horizon). The discrete actions
therefore optimise a horizon about three times shorter than the `size` component and the critic.
That is a deliberate part of the published rule, kept as published (#11); it applies only under
`disco103`.

## 6. Metrics

Each component logs under its trained module's learner metrics, with a `discorl_` prefix:

| Metric | Component |
|---|---|
| `discorl_target_kl` | `target_policy` |
| `discorl_value_norm_mean`, `discorl_value_norm_std` | `value_norm` |
| `discorl_policy_prediction_loss` | `policy_prediction` |
| `discorl_d103_<component>_{pi,y,z,aux,q}_loss`, `discorl_d103_size_policy_loss` | `disco103` |

`vf_explained_var` stays comparable with a plain PPO run's. With `value_norm` on it is computed
from normalised targets and predictions, and normalising both the same way leaves explained
variance unchanged.

`disco103` also logs PPO's own `mean_kl_loss` and `curr_kl_coeff`, for the components PPO still
trains (`size`), so the arm's KL column matches a plain PPO run's and Ray's adaptive coefficient
keeps moving.
