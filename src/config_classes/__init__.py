"""
Configuration classes for NORTRIP model.

This package contains dataclasses that define the structure for:
- Model flags
- Model parameters
- Model file paths
"""

from .model_flags import model_flags
from .model_parameters import model_parameters
from .model_file_paths import model_file_paths

__all__ = ["model_flags", "model_parameters", "model_file_paths"]
