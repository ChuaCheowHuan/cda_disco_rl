# Changelog

## 0.2.0 (unreleased)

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
