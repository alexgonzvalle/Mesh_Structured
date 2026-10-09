"""Structured grid construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import validate_positive


@dataclass(frozen=True)
class MeshDefinition:
    """Definition of a regular rectangular mesh."""

    xmin: float
    ymin: float
    dx: float
    dy: float
    nx: int
    ny: int

    @property
    def lx(self) -> float:
        """Total extent along x."""

        return (self.nx - 1) * self.dx

    @property
    def ly(self) -> float:
        """Total extent along y."""

        return (self.ny - 1) * self.dy

    def build_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the mesh coordinate arrays."""

        # xmax = self.xmin + self.lx
        # ymax = self.ymin + self.ly
        # x_axis = np.arange(self.xmin, xmax, self.dx, dtype=float)
        # y_axis = np.arange(self.ymin, ymax, self.dy, dtype=float)
        x_axis = self.xmin + (np.arange(self.nx) + 0.5) * self.dx
        y_axis = self.ymin + (np.arange(self.ny) + 0.5) * self.dy
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        return x_mesh, np.flipud(y_mesh)


def build_definition_from_bounds(
    *,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    dx: float,
    dy: float,
) -> MeshDefinition:
    """Create a mesh definition from bounding coordinates and spacing."""

    validate_positive(dx, name="dx")
    validate_positive(dy, name="dy")

    xmin = min(x1, x2)
    ymin = min(y1, y2)
    width = max(x1, x2) - xmin
    height = max(y1, y2) - ymin
    nx = round(width / dx)
    ny = round(height / dy)

    if nx <= 0 or ny <= 0:
        raise ValueError("Mesh bounds must define a positive area with at least one cell in each direction.")

    return MeshDefinition(xmin=xmin, ymin=ymin, dx=dx, dy=dy, nx=nx, ny=ny)
