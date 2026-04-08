# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DepthPredictor
from .val import DepthValidator

# Lazy import to avoid circular dependency (depth.train imports yolo.detect)
import importlib as _importlib


def __getattr__(name):
    if name == "DepthTrainer":
        return _importlib.import_module(".train", __name__).DepthTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = "DepthPredictor", "DepthTrainer", "DepthValidator"
