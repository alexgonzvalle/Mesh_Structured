from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def rectangular_plane_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.array([0.0, 0.5, 1.0])
    x_values, y_values = np.meshgrid(axis, axis)
    z_values = x_values + 2.0 * y_values
    return x_values, y_values, z_values
