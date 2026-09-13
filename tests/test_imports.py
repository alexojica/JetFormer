"""Every entry point must import in a fresh interpreter in any order (guards against import cycles)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

ORDERS = (
    "import jetformer.evaluation, jetformer.sampling, jetformer.sample, jetformer.train, jetformer.benchmark",
    "import jetformer.export, jetformer.training.checkpoint, jetformer.data",
    "import jetformer.sampling, jetformer.training.trainer, jetformer.evaluation",
    "import jetformer; from jetformer import Config, ConfigError, JetFormer, load_config",
)


@pytest.mark.parametrize("statement", ORDERS)
def test_modules_import_in_a_fresh_interpreter(statement):
    subprocess.run([sys.executable, "-c", statement], check=True, timeout=180)


def test_package_reports_its_installed_version():
    import jetformer

    assert jetformer.__version__ and jetformer.__version__ != "0.0.0+unknown"


def test_release_metadata_versions_agree():
    """The version is maintained by hand in two files; they must not drift apart."""
    pyproject, citation = ROOT / "pyproject.toml", ROOT / "CITATION.cff"
    if not pyproject.is_file() or not citation.is_file():  # pragma: no cover - installed package
        pytest.skip("release metadata is not part of an installed package")
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert declared == yaml.safe_load(citation.read_text())["version"]


@pytest.mark.parametrize("module", ["jetformer.train", "jetformer.sample", "jetformer.benchmark", "jetformer.export"])
def test_command_line_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], check=True, capture_output=True, text=True, timeout=180
    )
    assert "--set" in result.stdout
