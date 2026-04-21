# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Ultralytics utilities package."""

import contextlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Union

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ultralytics package root
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of Ultralytics multiprocessing threads
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
EMOJI = platform.system() != "Windows"  # emoji-safe logging
IMGSZ = 640  # default image size

# Logging
logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG if VERBOSE else logging.INFO,
)
LOGGER = logging.getLogger("ultralytics")


def is_colab() -> bool:
    """Check if the current environment is a Google Colab instance."""
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle() -> bool:
    """Check if the current environment is a Kaggle kernel."""
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter() -> bool:
    """Check if the current environment is a Jupyter Notebook."""
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False


def is_docker() -> bool:
    """Check if the current Python environment is running inside a Docker container."""
    with contextlib.suppress(Exception):
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    return False


def colorstr(*input) -> str:
    """Apply ANSI color codes to a string for terminal output.

    Args:
        *input: Variable length arguments. If multiple args, the last one is the string
                and the preceding ones are color/style names.

    Returns:
        str: The colored string.

    Example:
        >>> colorstr("blue", "bold", "Hello World!")
        '\033[34m\033[1mHello World!\033[0m'
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors.get(x, "") for x in args) + f"{string}" + colors["end"]


def get_ubuntu_version() -> Union[str, None]:
    """Return the Ubuntu version if the OS is Ubuntu, otherwise return None."""
    with contextlib.suppress(FileNotFoundError, AttributeError):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_ID"):
                    return line.strip().split("=")[-1].strip('"')
    return None


def get_user_config_dir(sub_dir: str = "Ultralytics") -> Path:
    """Return the appropriate config directory based on the operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create. Defaults to 'Ultralytics'.

    Returns:
        Path: The path to the user config directory.
    """
    if platform.system() == "Windows":
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif platform.system() == "Darwin":  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    else:  # Linux
        path = Path.home() / ".config" / sub_dir

    path.mkdir(parents=True, exist_ok=True)
    return path


USER_CONFIG_DIR = get_user_config_dir()
SETTINGS_YAML = USER_CONFIG_DIR / "settings.yaml"
