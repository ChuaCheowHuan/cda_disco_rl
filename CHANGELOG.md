# Changelog

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
- **Moved:** the package moved out of ChuaCheowHuan/disco_rl into its own repository, because it
  imports nothing from `disco_rl`.
