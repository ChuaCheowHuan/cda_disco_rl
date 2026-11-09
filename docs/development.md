# Development

## 1. Setup

gym-CDA is not on PyPI, so install it first, then this package in editable mode with its test
extra:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction@<ref>"
pip install -e '.[test]'
```

`<ref>` must be a gym-CDA commit at version 2.1.0 or later (§4), such as `master`; leave out
`@<ref>` to take `master`. Clone this repository with `--recurse-submodules` (§4) before the
editable install.

`pip install "cda-cbp @ git+https://github.com/ChuaCheowHuan/cda_cbp"` adds the tests that measure
a Continual Backprop reset against the target copies (`test_cbp_targets.py`); without it those
eight tests skip and the rest of the suite runs unchanged.

## 2. Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

The suite has 120 tests and takes a little over a minute on a CPU. Its files are:

| File | What it checks |
|---|---|
| `test_extension.py` | Registration through the real entry point, the defaults, every `validate` rule, and that a run not enabling `discorl` imports neither it nor JAX |
| `test_target_policy.py` | The EMA update of the target copy after each optimiser step, and that a copy that stopped matching the module fails loudly |
| `test_adv_ema.py` | The debiased moments against DiscoRL's formula, and the connector on selected modules and padding |
| `test_value_norm.py` | The module buffers, the cached `(mean, std)`, de-normalised `compute_values`, the loss on normalised targets, and the connector |
| `test_policy_prediction.py` | The discrete slices of the Dict action, the per-component KL with masked actions, and the connector that writes the next-step logits once per batch |
| `test_disco103.py` | The settings and weights from the installed `disco_rl`, the bridge's targets and carried state, the torch value loss against `disco_rl`'s own, the time-major packing, the jitted unroll against the eager one, PPO's KL term over the components Disco103 leaves to PPO, the rebuilt per-component distributions, and three real runs (local, a remote env runner, and `adv_ema`) |
| `test_cbp_targets.py` | What a Continual Backprop reset costs a target copy: the online-copy gap and the target-policy KL over the next 20 optimiser steps, for `target_policy` and for `disco103`. Skipped without cda-cbp |
| `test_packaging.py` | The installed `disco_rl` is the submodule unchanged, every subpackage of it is shipped, and the Disco103 weights are installed |
| `test_real_run.py` | Real gym-CDA runs: the extension enabled with every component off, every component but `disco103` together on `policy_0` with a remote env runner, the same components on a remote Learner, and an `lstm` encoder with the components it allows |

The component files test their pieces on stand-ins; the real runs show a change in gym-CDA's
Learner stack here rather than in a training run. `tests/conftest.py` holds the tiny run config,
one Ray session for the whole suite and `RealRun`, the base class that builds and trains an
`Algorithm` once per test class. An `Algorithm` build costs seconds, so add checks to an existing
run before adding a new one.

Style: `ruff format .` (line length 100) and `pyflakes cda_discorl tests`. CI does not check
these yet (#21).

## 3. CI

The workflow is [`.github/workflows/cda_discorl.yml`](../.github/workflows/cda_discorl.yml). It
runs on pushes to `main` and on pull requests: it installs CPU PyTorch, gym-CDA at
`GYM_CDA_REF` and this package, then runs the tests. Its checkout needs `submodules: true`, or
the build finds no `disco_rl`.

`GYM_CDA_REF` defaults to gym-CDA's `master`. A manual run can test another ref with the
`gym_cda_ref` input.

## 4. Compatibility with gym-CDA

The package depends on `gym_continuousDoubleAuction>=2.1,<3`. Version 2.1.0 is the first that
lists extensions without importing them, so a run that does not enable `discorl` never imports this
package or JAX, and the first that reads `extra_trunk_heads`.

What the package relies on in gym-CDA:

- the `Extension` dataclass and `settings_for` in `train/extensions.py`
- `CDAPPOTorchLearner._policy_value_loss`, the seam around PPO's loss
- `POLICY_PREFIX` from `train/policy/policy_handler.py`
- `_collect` from `train/model/moe_learner.py`, which `policy_prediction` and `disco103` use to
  discard the MoE routing statistics of their extra forward passes. It is private (#26)
- `extra_trunk_heads`, read by `train/model/layout.py`, which declares `aux_pi` and `disco_heads`
  as heads of the trunk `pi` reads

What the package imports from `disco_rl` is listed in [architecture.md](architecture.md) §5.
`disco_rl` is the git submodule `third_party/disco_rl` (the ChuaCheowHuan/disco_rl fork, pinned
and unchanged), so clone with `--recurse-submodules` and check out CI with `submodules: true`.
To move to a newer fork commit, check it out in the submodule, commit the new pin and run the
tests. `pyproject.toml` lists the subpackages shipped, the weights file and the fork's runtime
dependencies; a fork commit that adds a subpackage or a dependency needs them added there.
ruff excludes `third_party`, so the fork's code is never reformatted.

gym-CDA's own tests of the contract are `test_extensions.py` and
`integration/test_extensions_wiring.py`.

## 5. Adding a component

1. Import from `disco_rl` (through `upstream.py`) anything it already provides: functions,
   settings, defaults. Write locally only what must be torch, or what is specific to gym-CDA.
2. Write the mixins in a new module. Check the component's switch in `build`, `setup` or the loss,
   and call `super()` in every overridden method.
3. Add its keys to `DEFAULTS`, its switch to `COMPONENTS`, and its rules to `validate` in
   `config.py`. If it keeps Learner state, add it to `PER_LEARNER_STATE`. If it adds tensors to
   the module, add it to `STRUCTURAL_KEYS` and `components.MODULE_COMPONENTS`.
4. Add its mixins to `DiscoLearnerMixin` and `DiscoModuleMixin` in `components.py`.
5. Add a test file for its pieces, and add the component to the combined run in
   `test_real_run.py` (with its checks there) unless it excludes another component.
6. Document it in the README, [components.md](components.md), [configuration.md](configuration.md),
   [architecture.md](architecture.md) and the [changelog](../CHANGELOG.md).

## 6. Releasing

Bump `version` in `pyproject.toml` and `__version__` in `cda_discorl/__init__.py` together.
gym-CDA records this version beside every checkpoint that used the extension.
