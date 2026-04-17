# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.0"

from ultralytics.models import YOLO, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "SETTINGS",
    "YOLO",
    "YOLOWorld",
    "checks",
    "download",
)
