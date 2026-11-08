"""What the suite shares: one Ray session, the tiny run config and a real-run base class.

A real run is the expensive part of the suite (an `Algorithm` build is seconds), so each test
class that needs one subclasses `RealRun`, which builds and trains it once for the class.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile

import pytest
import ray

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

#: The smallest run that still trains: four agents, two of them trained, 64-step episodes.
TINY = dict(
    num_agents=4,
    num_trained_agents=2,
    max_step=64,
    num_episodes_per_iter=2,
    num_epochs=1,
    minibatch_size=64,
    num_env_runners=0,
    num_learners=0,
    num_gpus_per_learner=0,
    episode_data_dir=None,
    log_level="ERROR",
)


def train_config(settings=None, **overrides) -> TrainConfig:
    """A tiny `TrainConfig` with `discorl` enabled and `settings` as its block."""
    return TrainConfig(
        **{**TINY, **overrides},
        extensions_enabled=["discorl"],
        extension_settings={"discorl": settings or {}},
    )


@pytest.fixture(scope="session")
def ray_session():
    # Three CPUs: a run with one remote env runner, plus the runner of its restored copy.
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=3)
    yield
    ray.shutdown()


class RealRun:
    """Base for a test class that trains one real gym-CDA `Algorithm` with `discorl` on.

    Set `settings` (the `discorl` block), `overrides` (train-config keys) and `iterations`.
    The class then has `cfg`, `algo`, `learner` (the local Learner), `results` (one per
    iteration) and `result` (the last). Override `before_training` or `train_once` to
    observe the run.
    """

    settings: dict = {}
    overrides: dict = {}
    iterations = 1

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _real_run(cls, ray_session):
        extensions.discover.cache_clear()
        log_dir = tempfile.mkdtemp()
        cls.cfg = train_config(cls.settings, seed=7, log_base_dir=log_dir, **cls.overrides)
        ppo_config, _callbacks = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.learner = cls.algo.learner_group._learner
        try:
            cls.before_training()
            cls.results = [cls.train_once() for _ in range(cls.iterations)]
            cls.result = cls.results[-1]
            yield
        finally:
            cls.algo.stop()
            shutil.rmtree(log_dir, ignore_errors=True)

    @classmethod
    def before_training(cls) -> None:
        pass

    @classmethod
    def train_once(cls):
        return cls.algo.train()

    @contextlib.contextmanager
    def restored(self):
        """The Learner of an `Algorithm` restored from a checkpoint of this run."""
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            yield restored.learner_group._learner
        finally:
            restored.stop()


def assert_declared_on_the_pi_trunk(module, head) -> None:
    """`head` reads the actor embedding, so it must be an extra head of the trunk `pi` reads.

    Continual Backprop resets the input weights of the heads `trunk_outputs` reports, so a
    head it did not see would read a replaced unit with the old unit's weights (#5).
    """
    from gym_continuousDoubleAuction.train.model.layout import trunk_outputs

    trunks = trunk_outputs(module)
    reads_pi = [t for t in trunks if module.pi in t.heads]
    assert reads_pi and all(t.extra_heads == [head] for t in reads_pi)
    assert all(t.extra_heads == [] for t in trunks if module.pi not in t.heads)
