"""`disco_rl` ships from the pinned submodule, unchanged, with the Disco103 weights.

The fork is never edited, so `pyproject.toml` does its packaging: it maps the submodule to the
top-level package `disco_rl`, lists its subpackages and adds the weights as package data.
"""

import hashlib
import pathlib
import tomllib
from importlib import resources

import pytest
import setuptools

import disco_rl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "third_party" / "disco_rl"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]

needs_submodule = pytest.mark.skipif(
    not (SUBMODULE / "disco_rl").is_dir(), reason="submodule not checked out"
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_weights_are_installed():
    weights = resources.files("disco_rl.update_rules") / "weights" / "disco_103.npz"
    assert weights.is_file()


@needs_submodule
def test_every_subpackage_of_the_fork_is_shipped():
    found = setuptools.find_packages(SUBMODULE, include=["disco_rl", "disco_rl.*"])
    shipped = [p for p in PYPROJECT["packages"] if p.split(".")[0] == "disco_rl"]
    assert sorted(found) == sorted(shipped)
    assert PYPROJECT["package-dir"] == {"disco_rl": "third_party/disco_rl/disco_rl"}


@needs_submodule
def test_the_installed_code_is_the_submodule_unchanged():
    installed = pathlib.Path(disco_rl.__file__).parent
    source = SUBMODULE / "disco_rl"
    files = sorted(p.relative_to(source) for p in source.rglob("*.py"))
    files.append(pathlib.Path("update_rules/weights/disco_103.npz"))
    assert files
    for rel in files:
        assert _digest(installed / rel) == _digest(source / rel), rel
