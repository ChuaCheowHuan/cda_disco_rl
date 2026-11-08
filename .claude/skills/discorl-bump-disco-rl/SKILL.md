---
name: discorl-bump-disco-rl
description: Move the pinned disco_rl submodule (third_party/disco_rl, the ChuaCheowHuan/disco_rl fork) to a newer commit and keep the packaging, dependencies and imports in step. Use for "update disco_rl", "bump the fork" or when upstream DeepMind changes land in the fork.
---

# Move the disco_rl pin

The fork is installed unchanged as the top-level package `disco_rl`, with its subpackages and
the Disco103 weights listed by hand in `pyproject.toml`. A new commit can add a subpackage, a
dependency or an API change, and none of that is picked up automatically.

## Steps

1. **Pick the commit** on ChuaCheowHuan/disco_rl and read its diff against the current pin
   (`git -C third_party/disco_rl log --oneline <old>..<new>`). Note new subpackages, new
   imports in shipped code, and changes to anything `cda_discorl` imports: `agent.get_settings_disco`,
   `update_rules.disco.DiscoUpdateRule` (its `__init__` defaults, `unroll_meta_net`,
   `init_params`, `_adv_ema`/`_td_ema`), `utils.MovingAverage`, `utils.batch_lookup`,
   `value_fns.value_utils`, `types.EmaState`, `types.UpdateRuleInputs`, and the weights file.
2. **Move the pin**: `git -C third_party/disco_rl fetch && git -C third_party/disco_rl checkout <new>`,
   then `git add third_party/disco_rl`. Never edit files inside the submodule.
3. **Packaging**: add any new subpackage to `[tool.setuptools] packages` and any new runtime
   import to `dependencies` in `pyproject.toml` (leave out the colab-only plotting ones).
4. **Reinstall and test**: `pip install -e '.[test]'`, then `OMP_NUM_THREADS=1 python -m pytest -q`.
   `test_packaging.py` fails on a missing subpackage or a changed installed file;
   `TestDefaultsFromDiscoRL` and `TestBridge` fail when the published settings, costs or weights
   moved. If a published default changed, that changes runs: say so in the changelog.
5. **Docs**: update `docs/architecture.md` §5 if what is imported changed, and add a changelog
   entry naming the old and new fork commits.
6. Commit the pin, the packaging and the docs together, one commit per pin move.
