"""JetFormer: a PyTorch implementation of the Jet normalizing flow and the JetFormer image model."""

from importlib.metadata import PackageNotFoundError, version

from jetformer.config import Config, ConfigError, load_config
from jetformer.model.jetformer import JetFormer

try:
    __version__ = version("jetformer")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["Config", "ConfigError", "JetFormer", "__version__", "load_config"]
