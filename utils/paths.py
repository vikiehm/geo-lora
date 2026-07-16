"""Loader for the machine-specific paths config (config/paths.yaml).

Every cluster-/machine-specific value (dataset roots, DINOv3 weights, checkpoint
folder, W&B settings) lives in config/paths.yaml so the rest of the code stays
path-free. Copy config/paths.example.yaml to config/paths.yaml and edit it.
"""

import os
import functools

import yaml

# Repo root = parent of this file's directory (utils/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATHS = os.path.join(_REPO_ROOT, "config", "paths.yaml")
_EXAMPLE_PATHS = os.path.join(_REPO_ROOT, "config", "paths.example.yaml")


@functools.lru_cache(maxsize=None)
def load_paths(path=_DEFAULT_PATHS):
    """Load and cache the paths config. Raises a helpful error if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Paths config not found at {path}.\n"
            f"Copy the template and edit it for your machine:\n"
            f"    cp {os.path.relpath(_EXAMPLE_PATHS, _REPO_ROOT)} "
            f"{os.path.relpath(path, _REPO_ROOT)}"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def data_path(*relative):
    """Join one or more dataset-relative path components onto data_root."""
    return os.path.join(load_paths()["data_root"], *relative)


def checkpoint_path(*relative):
    """Join one or more components onto save_folder (where checkpoints live)."""
    return os.path.join(load_paths()["save_folder"], *relative)
