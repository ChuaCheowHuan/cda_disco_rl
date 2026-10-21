# Architecture

cda-discorl is a gym-CDA *extension*: a separately installed package that gym-CDA finds through
an entry point and composes into a run only when the run asks for it. This page explains how the
pieces fit together. gym-CDA's side of the contract is its
[docs/30_extensions.md](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/docs/30_extensions.md).

## 1. Registration

`pyproject.toml` registers one object:

```toml
[project.entry-points."gym_cda.extensions"]
discorl = "cda_discorl:EXTENSION"
```

`cda_discorl/__init__.py` builds that `Extension` from four pieces:

| Field | Value | Role |
|---|---|---|
| `defaults`, `validate` | `config.DEFAULTS`, `config.validate` | Every setting and the rules between them ([configuration.md](configuration.md)) |
| `learner_mixin` | `components.DiscoLearnerMixin` | Loss terms, connectors and Learner state |
| `module_mixin`, `module_ids` | `components.DiscoModuleMixin`, `components.module_ids` | Extra buffers and heads, only on the selected modules |
| `env_wrapper` | `env_variety.wrap` | The per-episode market draw |

`structural_keys` lists `value_norm` and `policy_prediction`, the two settings that add tensors to
a module. gym-CDA refuses a resume that changes either.

## 2. How the mixins compose

gym-CDA allows one Learner mixin and one module mixin per extension. Each component brings its
own mixin, and `components.py` stacks them:

```python
class DiscoLearnerMixin(ValueNormLearner, TargetPolicyLearner, PolicyPredictionLearner, AdvEmaLearner): ...
class DiscoModuleMixin(ValueNormModule, PolicyPredictionModule): ...
```

Every mixin checks its own switch and does nothing when it is off. Each overridden method calls
`super()`, so all of them run in that order and then reach gym-CDA's Learner. `ValueNormLearner`
comes first so that every later loss term sees the normalised value targets.

gym-CDA places the Learner mixin after the encoder's Learner and before Continual Backprop. So the
MoE and JEPA loss terms are kept, and CBP still wraps the result.

`module_ids` returns no modules unless `value_norm` or `policy_prediction` is on. A run that uses
only Learner-side components therefore leaves every module class exactly as gym-CDA builds it.

## 3. The hooks each component uses

All loss changes go through `_policy_value_loss`, gym-CDA's seam around PPO's own loss. Overriding
it, rather than `compute_loss_for_module`, keeps the MoE and JEPA terms.

| Component | Learner hooks | Module hooks | Learner connector | State saved |
|---|---|---|---|---|
| `target_policy` | `build` deep-copies each selected module; `_policy_value_loss` adds `target_kl_cost * KL(pi_target ‖ pi)`; `apply_gradients` moves the copy `1 - target_coeff` toward the online weights | – | – | Target weights, in the Learner state under `discorl_target_policy` |
| `adv_ema` | `build` inserts the connector | – | `EmaAdvantages`, after `GeneralAdvantageEstimation` | EMA moments, in the Learner state under `discorl_adv_ema` |
| `value_norm` | `build` inserts the connector; `_policy_value_loss` swaps in normalised value targets and reads the raw head | `setup` adds float64 moment buffers; `compute_values` maps the head back to reward units | `ValueNormMoments`, after `GeneralAdvantageEstimation` | Moment buffers, with the module weights; `set_state` restores them as float64, since RLlib's cast to float32 would blur the std |
| `env_variety` | – | – | – | None; a seeded `reset` repeats its draw |
| `policy_prediction` | `build` adds next observations to the batch if the run lacks them; `_policy_value_loss` adds the prediction KL | `setup` adds the `aux_pi` head (Learner copy only); `_forward_train` fills it | `AddNextObservationsFromEpisodesToTrainBatch`, when missing | The head's weights, with the module |

### Advantage EMA

RLlib's GAE connector writes standardised advantages, and the raw values are gone by the time it
finishes. `EmaAdvantages` therefore recomputes them as `value_targets - V(s)` with the same
weights. It then folds the batch into a debiased EMA of the first and second moments and
normalises with that.

### Value normalisation

`ValueNormMoments` updates the module's moments from the masked `VALUE_TARGETS` right after GAE,
or right after `EmaAdvantages` when `adv_ema` is on too. That order matters: `EmaAdvantages`
recomputes `V(s)` through `compute_values`, and it must use the stats GAE bootstrapped with, not
ones already updated from this batch.
In the loss, `ValueNormLearner` replaces the targets with `(T - mean) / std` and sets the
module's `_vn_raw` flag around the `super()` call, so PPO's critic loss compares normalised
targets with the head's normalised output. Everywhere else, `compute_values` returns
`mean + std * head`, so GAE and RLlib's metrics keep reward units.

### Env variety

`wrap` replaces the env's `reset`. It draws one choice per key, applies it through the env's own
setters, and records it on `env.discorl_env_settings`. Only settings that change neither the
observation nor the action space are allowed, because gym-CDA builds the spaces before any
wrapper runs.

## 4. Checkpoints and resumes

The Learner-side state (`discorl_target_policy`, `discorl_adv_ema`) travels in `get_state` and
`set_state`, and the module-side state travels with the weights. gym-CDA records the package
version beside each checkpoint. It refuses to restore one that used `discorl` where the package
is not installed, and refuses a resume whose settings differ from the checkpoint's.

The three stateful components refuse `num_learners > 1`. With several Learners, each would keep
its own copy of the state, fitted to its own shard of the batch.

## 5. Where each mechanism comes from

The components are PyTorch reimplementations, and the package imports nothing from
[google-deepmind/disco_rl](https://github.com/google-deepmind/disco_rl), which is JAX. Each
module's docstring names the upstream file:

| Component | Upstream |
|---|---|
| `adv_ema`, `value_norm` | `disco_rl/utils.py` (`MovingAverage`), `disco_rl/value_fns` |
| `target_policy` | `target_params` and `target_params_coeff` in `disco_rl/update_rules/disco.py` |
| `policy_prediction` | `aux_pi` in `disco_rl/update_rules/disco.py` |
| `env_variety` | The paper's training across many environments; there is no upstream code |
