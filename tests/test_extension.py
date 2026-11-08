"""The package as gym-CDA sees it: installed, discovered, validated, and inert.

These run against an installed gym-CDA, through its public extension point, so
a change there that breaks this package fails here.
"""

from types import SimpleNamespace

import pytest

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.model.moe_learner import CDAPPOTorchLearner
from gym_continuousDoubleAuction.train.train import TrainConfig, _learner_class

import cda_discorl
from cda_discorl import config

from conftest import TINY, train_config as _cfg


class TestDiscovery:
    def test_the_entry_point_resolves_to_this_package(self):
        extensions.discover.cache_clear()
        assert extensions.discover()["discorl"] is cda_discorl.EXTENSION

    def test_the_recorded_version_is_the_package_version(self):
        from importlib.metadata import version

        assert cda_discorl.EXTENSION.version == version("cda-discorl")


class TestEnabledWithEverythingOffChangesNothing:
    def test_every_component_defaults_off(self):
        assert not any(config.DEFAULTS[name] for name in config.COMPONENTS)

    def test_the_learner_only_gains_the_switched_mixin(self):
        composed = _learner_class(_cfg())
        assert composed.__mro__[1] is cda_discorl.EXTENSION.learner_mixin
        assert composed.__mro__[-len(CDAPPOTorchLearner.__mro__) :] == CDAPPOTorchLearner.__mro__

    def test_no_module_gets_the_module_mixin(self):
        assert cda_discorl.EXTENSION.module_ids(config.DEFAULTS, ["policy_0", "policy_1"]) == []

    def test_the_env_is_unchanged(self):
        env = SimpleNamespace(reset=lambda **kw: None)
        reset = env.reset
        assert extensions.wrap_env(env, ["discorl"], {}) is env
        assert env.reset is reset

    def test_the_settings_reach_the_learner_config(self):
        group = {extensions.EXTENSIONS_CONFIG_KEY: extensions.learner_config(_cfg())}
        assert extensions.settings_for(group, "discorl") == config.DEFAULTS
        assert extensions.versions(group) == {"discorl": cda_discorl.__version__}


class TestValidation:
    @pytest.mark.parametrize("key", ["target_kl", "env_variety"])
    def test_an_unknown_key_is_refused_by_gym_cda(self, key):
        # env_variety went in 0.2.0 in favour of gym-CDA's own `market_variety` env key.
        with pytest.raises(ValueError, match="unknown keys"):
            _cfg({key: True})

    def test_a_component_not_built_yet_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "BUILT", frozenset(config.COMPONENTS) - {"disco103"})
        with pytest.raises(ValueError, match="not built yet"):
            _cfg({"disco103": True})

    def test_modules_must_be_trainable(self):
        with pytest.raises(ValueError, match="not trainable"):
            _cfg({"modules": ["policy_3"]})

    def test_modules_may_not_be_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _cfg({"modules": []})

    def test_both_trainable_modules_may_be_selected(self):
        _cfg({"modules": ["policy_0", "policy_1"]})

    def test_a_switch_must_be_a_bool(self):
        with pytest.raises(ValueError, match="true or false"):
            _cfg({"adv_ema": 1})

    @pytest.mark.parametrize(
        "key, value",
        [
            ("target_coeff", 1.0),
            ("target_kl_cost", -0.1),
            ("policy_prediction_cost", -1.0),
            ("adv_ema_decay", -0.1),
            ("value_norm_decay", 1.0),
            ("disco103_reward_scale", 0.0),
        ],
    )
    def test_out_of_range_values_are_refused(self, key, value):
        with pytest.raises(ValueError, match=key):
            _cfg({key: value})

    @pytest.mark.parametrize("name", ["policy_prediction", "disco103"])
    def test_a_recurrent_encoder_is_refused_where_the_next_step_is_needed(self, name):
        with pytest.raises(ValueError, match="stateless"):
            _cfg({name: True}, encoder_type="lstm")


class TestCombinationRules:
    """Checked on `validate` directly, without gym-CDA's config build."""

    @staticmethod
    def _check(settings, **cfg_kwargs):
        cfg = TrainConfig(**{**TINY, **cfg_kwargs})
        config.validate({**config.DEFAULTS, **settings}, cfg)

    @pytest.mark.parametrize("name", config.PER_LEARNER_STATE)
    def test_per_learner_state_refuses_several_learners(self, name):
        with pytest.raises(ValueError, match="num_learners"):
            self._check({name: True}, num_learners=2)

    def test_every_component_but_disco103_together_is_allowed(self):
        self._check({name: True for name in config.COMPONENTS if name != "disco103"})

    @pytest.mark.parametrize("name", config.DISCO103_EXCLUDES)
    def test_disco103_refuses_the_components_it_excludes(self, name):
        with pytest.raises(ValueError, match="cannot be combined"):
            self._check({"disco103": True, name: True})

    def test_disco103_combines_with_the_rest(self):
        rest = set(config.COMPONENTS) - set(config.DISCO103_EXCLUDES)
        self._check({name: True for name in rest})


class TestDefaultsFromDiscoRL:
    """Defaults DiscoRL defines are imported from it, not restated here."""

    def test_they_are_the_published_values(self):
        from disco_rl import agent

        hp = agent.get_settings_disco().hyper_params
        assert config.DEFAULTS["target_coeff"] == hp.target_params_coeff
        assert config.DEFAULTS["policy_prediction_cost"] == hp.aux_policy_cost

    def test_the_decays_are_the_update_rules(self):
        import inspect

        from disco_rl.update_rules.disco import DiscoUpdateRule

        want = inspect.signature(DiscoUpdateRule).parameters["moving_average_decay"].default
        assert config.DEFAULTS["adv_ema_decay"] == config.DEFAULTS["value_norm_decay"] == want


class TestResume:
    def test_structural_settings_are_registered(self):
        group = {extensions.EXTENSIONS_CONFIG_KEY: extensions.learner_config(_cfg())}
        assert set(extensions.structural_keys(group)) == {
            f"ext.discorl.{k}" for k in config.STRUCTURAL_KEYS
        }


class TestInstalledButNotEnabled:
    """Installing this package costs a run that does not enable it nothing (#4).

    The package imports JAX and `disco_rl` (its defaults come from there), so
    gym-CDA must not import it until a run enables `discorl`: gym-CDA 2.1 lists
    entry points without loading them (gym-continuousDoubleAuction#337).
    """

    def test_a_default_train_config_imports_neither_jax_nor_this_package(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from gym_continuousDoubleAuction.train.train import TrainConfig\n"
            "from gym_continuousDoubleAuction.train import extensions\n"
            "assert 'discorl' in extensions.available()\n"
            "TrainConfig(episode_data_dir=None, log_level='ERROR')\n"
            "print(sorted(m for m in ('jax', 'disco_rl', 'cda_discorl') if m in sys.modules))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == "[]"
