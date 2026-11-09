# Changelog

## 0.2.0 (unreleased)

- **Docs corrected:** `docs/development.md` now says what CI does (it checks formatting and
  pyflakes, and runs a `v2.3.0` gate beside a non-blocking `master` job) and gives the suite's
  current size (120 tests, about a minute and a half); `CLAUDE.md` matches.
- **Fixed:** `validate` refuses a `modules` list that names a module twice (#12). Several pieces
  loop over the list as given, so a repeat updated `value_norm` and Disco103 state twice per
  batch, and `ValueNormMoments` decayed its EMA twice.
- **Fixed:** `validate` refuses a numeric setting that is not a finite number (#13), with the
  package's `extension_settings['discorl']: ...` message. A string used to raise a bare
  `TypeError` from the comparison, a `true`/`false` passed as 0/1, and `inf` passed the
  non-negative checks. `BUILT` stays as the extension point it always was, now documented.
- **Fixed:** with `disco103` on, the continuous `size` component keeps PPO's KL penalty (#10).
  Ray's `use_kl_loss` defaults to true, so plain PPO trains `size` with `kl_coeff * KL(prev ||
  curr)` and an adaptive coefficient, while the Disco103 arm dropped that term. The arm now adds
  it for the non-discrete components and logs `mean_kl_loss`, so Ray's coefficient keeps moving.
- **Fixed:** `ema_update_` pairs the target and online parameters by name with a strict `zip`, and
  raises when the copy no longer matches the module (#14). A target that lost or gained a head
  used to blend the wrong tensors, or stop early, without a word.
- **Fixed:** Disco103 checks that the batch rows are the episodes it was handed, and that the
  masked rows are the bootstrap rows, before the rule sees them (#16). A Ray change that
  reordered or re-chunked rows now fails loudly instead of training on shifted trajectories.

- **Performance:** the Disco103 meta-network unroll is jitted (#29), one compilation per rollout
  length. A 512 × 4 rollout takes about 0.04 s against 0.28 s eagerly; the fork stays unedited, a
  shim answers the one `isinstance(logits, chex.Array)` guard for the tracer.
- **Performance:** `value_norm` caches its `(mean, std)` and refreshes it only when the moments
  move (#30), instead of two JAX `normalize` calls on every `compute_values`.
- **Changed:** `policy_prediction` computes the next step's logits once per update batch, in a
  connector right after GAE, and every minibatch reads them (#31). They are a pre-update target,
  as in DiscoRL; with `num_epochs > 1` the loss therefore trains toward the pre-update policy
  rather than toward the policy after earlier epochs' updates.
- **Refactor:** Disco103's `size` loss rebuilds each component's distribution from its slice of
  `ACTION_DIST_INPUTS` and no longer reads Ray's private `TorchMultiDistribution._original_struct`
  (#24); the bridge builds its EMA states with `disco_rl.utils.MovingAverage` and the public
  settings instead of the rule's private `_adv_ema` / `_td_ema` (#25).

- **CI:** the workflow has a `permissions: contents: read` block, the JAX stack carries upper
  bounds, and the gate runs against the pinned gym-CDA `v2.3.0` tag with a non-blocking `master`
  job beside it (#20). A `lint` step runs `ruff format --check .` and `pyflakes cda_discorl tests`
  before the tests (#21).
- **Docs:** `docs/components.md` and `docs/findings.md` record that Disco103 keeps its published
  `value_discount` 0.997 while the rest of the Learner uses the run's `gamma` (#11, #23).
- **Tests:** real runs for a remote env runner under Disco103, `disco103` with `adv_ema`, a remote
  Learner, and an `lstm` encoder with `target_policy`, `adv_ema` and `value_norm` (#19); and a
  measurement of a Continual Backprop reset against both kinds of target copy (#27).

- **Tests:** the per-component real runs became one run with every component but `disco103`
  together, sharing a Ray session and a `RealRun` base class (`tests/conftest.py`); redundant and
  trivial tests went. 103 tests in about three minutes became 81 in a little over one.
- **Added:** `CLAUDE.md` and the project skills `discorl-verify-finding` and
  `discorl-bump-disco-rl`.

- **Docs:** each component's description moved from the README into `docs/components.md`, with
  the metrics table that was in gym-CDA's docs/30; the README keeps the summary table and gains a
  diagram of the fork, this package and gym-CDA. The README's market-variety section and test
  section went: the first is `docs/configuration.md` §3, the second `docs/development.md`.

- **Added:** `disco103`, the frozen Disco103 rule in place of PPO's loss on the discrete action
  components (experimental), and `disco103_reward_scale`. It imports the rule, its settings and
  its weights from `disco_rl`; none of them are copied here.
- **Changed:** `disco_rl` is now a dependency, so the package brings JAX, Haiku and rlax. It is
  the ChuaCheowHuan/disco_rl fork as the pinned git submodule `third_party/disco_rl`, installed
  unchanged as the top-level package `disco_rl` with the Disco103 weights mapped in at build time;
  the fork is not edited.
- **Removed:** `env_variety`, with `env_variety_keys` and `env_variety_choices`. The per-episode
  market draw is now gym-CDA's env key `market_variety`
  (ChuaCheowHuan/gym-continuousDoubleAuction#334), so the package no longer wraps the env. A
  settings block that still names one of the three keys is refused as unknown; move the choices
  to `market_variety` in the env or train config. Anything `disco_rl` provides is imported, not reimplemented:
  `adv_ema` and `value_norm` use `disco_rl.utils.MovingAverage`; the defaults of `target_coeff`,
  `policy_prediction_cost`, `adv_ema_decay` and `value_norm_decay` are read from `disco_rl`
  (same values as before). `MovingAverage` computes in float32, where the old code used float64.
  Checkpoints from 0.1.0 load unchanged: the moments are stored as the same three numbers.
- **Changed:** needs gym-CDA 2.1 (`gym_continuousDoubleAuction>=2.1,<3`). 2.1.0 lists extensions
  without importing them, so a run that does not enable `discorl` no longer imports this package
  or JAX (#4, #6).
- **Changed:** `PolicyPredictionModule` and `Disco103Module` declare their heads (`aux_pi`,
  `disco_heads`) in gym-CDA's `extra_trunk_heads`, as heads of the trunk `pi` reads, so Continual
  Backprop (cda-cbp) resets their input weights with a unit it replaces (#5).

## 0.1.0 (unreleased)

First version, as a gym-CDA extension named `discorl`.

- **Added:** five components, each off by default and applied to the modules listed in
  `modules`:
  - `target_policy`
  - `adv_ema`
  - `value_norm`
  - `env_variety`
  - `policy_prediction` (experimental)
- **Not included:** DiscoRL's categorical critic and the frozen Disco103 update rule. An earlier,
  unreleased version on ChuaCheowHuan/disco_rl carried both, and neither learned in gym-CDA's
  market ([docs/findings.md](docs/findings.md)).
- **Moved:** the package moved out of ChuaCheowHuan/disco_rl into its own repository. (It imported
  nothing from `disco_rl` then; 0.2.0 does.)
