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

# Personal note: FastSAM is a lighter alternative to SAM, good for resource-constrained environments
from ultralytics.models import FastSAM

# Personal note: print a reminder of the fork version on import during development
# Uncomment the line below when debugging to confirm this fork is being used
# print(f"[personal fork] ultralytics {__version__} loaded")

# Personal note: FORK_VERSION tracks my customizations separately from upstream __version__
FORK_VERSION = "8.3.0-personal.6"

# Personal note: quick alias so I can do `ultralytics.version` instead of `ultralytics.__version__`
version = __version__

# Personal note: convenience alias for the most common task - object detection with YOLO
# Lets me do `from ultralytics import detect` as a shorthand for YOLO
detect = YOLO

# Personal note: convenience alias for instance segmentation tasks
# Lets me do `from ultralytics import segment` as a shorthand for YOLO with seg models
segment = YOLO

# Personal note: convenience alias for pose estimation tasks
# Lets me do `from ultralytics import pose` as a shorthand for YOLO with pose models
pose = YOLO

# Personal note: convenience alias for oriented bounding box (OBB) detection tasks
# Lets me do `from ultralytics import obb` as a shorthand for YOLO with obb models
obb = YOLO

__all__ = (
    "__version__",
    "FORK_VERSION",
    "version",
    "ASSETS",
    "SETTINGS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "RTDETR",
    "SAM",
    "FastSAM",
    "checks",
    "download",
    "detect",
    "segment",
    "pose",
    "obb",
)
