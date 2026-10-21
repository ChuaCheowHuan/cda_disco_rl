# Development

## 1. Setup

gym-CDA is not on PyPI, so install it first, then this package in editable mode with its test
extra:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction@<ref>"
pip install -e '.[test]'
```

`<ref>` must be a gym-CDA commit that has the extension point, `train/extensions.py`. Until
ChuaCheowHuan/gym-continuousDoubleAuction#314 merges, that is its branch,
`claude/discorl-fresh`. After it merges, use `master` or the first release tag that carries it.

## 2. Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

The suite takes about a minute on a CPU. Its tests are:

| File | What it checks |
|---|---|
| `test_extension.py` | Registration through the real entry point, the defaults, and every `validate` rule |
| `test_target_policy.py` | The KL term, the EMA update after each step, and the Learner-state round trip |
| `test_adv_ema.py` | The debiased moments, the connector's placement after GAE, and the state round trip |
| `test_value_norm.py` | The module buffers, de-normalised `compute_values`, the loss on normalised targets, a checkpoint round trip, and its order after `EmaAdvantages` when `adv_ema` is on too |
| `test_env_variety.py` | The per-episode draw, seeding, and the fee rules |
| `test_policy_prediction.py` | The head's shape, the per-component KL, terminal masking, and the next-observation connector |
| `test_real_run.py` | A real gym-CDA `Algorithm` training with every component on, with the extension recorded in its checkpoint |

Each component's file also builds a real Learner where it matters, so a change in gym-CDA's
Learner stack shows up here rather than in a training run.

Style: `ruff format .` (line length 100) and `pyflakes cda_discorl tests`.

## 3. CI

The workflow is [`.github/workflows/cda_discorl.yml`](../.github/workflows/cda_discorl.yml). It
runs on pushes to `main` and on pull requests: it installs CPU PyTorch, gym-CDA at
`GYM_CDA_REF` and this package, then runs the tests.

`GYM_CDA_REF` defaults to gym-CDA's `claude/discorl-fresh` branch. Point it at `master` or a
release tag once the extension point has merged. A manual run can override it with the
`gym_cda_ref` input.

## 4. Compatibility with gym-CDA

The package depends on `gym_continuousDoubleAuction>=2.0,<3`. Version 2.0.0 itself predates the
extension point, so the lower bound should rise to the first gym-CDA release that carries it.

What the package relies on in gym-CDA:

- the `Extension` dataclass and `settings_for` in `train/extensions.py`
- `CDAPPOTorchLearner._policy_value_loss`, the seam around PPO's loss
- the env's `matching_rule`, `step_clearing`, `maker_fee_bps` and `taker_fee_bps` attributes, and
  `validate_fee_rates`
- `POLICY_PREFIX` from `train/policy/policy_handler.py`
- `_collect` from `train/model/moe_learner.py`, which `policy_prediction` uses to discard the MoE
  routing statistics of its extra forward pass

gym-CDA's own tests of the contract are `test_extensions.py` and
`integration/test_extensions_wiring.py`.

## 5. Adding a component

1. Write the mixins in a new module. Check the component's switch in `build`, `setup` or the loss,
   and call `super()` in every overridden method.
2. Add its keys to `DEFAULTS`, its switch to `COMPONENTS`, and its rules to `validate` in
   `config.py`. If it keeps Learner state, add it to `PER_LEARNER_STATE`. If it adds tensors to
   the module, add it to `STRUCTURAL_KEYS` and `components.MODULE_COMPONENTS`.
3. Add its mixins to `DiscoLearnerMixin` and `DiscoModuleMixin` in `components.py`.
4. Add a test file, and check that `test_real_run.py` still passes with every component on.
5. Document it in the README, [configuration.md](configuration.md),
   [architecture.md](architecture.md) and the [changelog](../CHANGELOG.md).

## 6. Releasing

Bump `version` in `pyproject.toml` and `__version__` in `cda_discorl/__init__.py` together.
gym-CDA records this version beside every checkpoint that used the extension.
