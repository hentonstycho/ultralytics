# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.0"

from ultralytics.models import YOLO, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

# Personal fork: also expose NAS and RTDETR models at top level for convenience
from ultralytics.models import NAS, RTDETR

# Personal note: SAM is also useful for segmentation tasks
from ultralytics.models import SAM

__all__ = (
    "__version__",
    "ASSETS",
    "SETTINGS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "RTDETR",
    "SAM",
    "checks",
    "download",
)
