"""Public mesh API for structured bathymetry workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import utm
from matplotlib.axes import Axes
from matplotlib.path import Path as MatplotlibPath
from scipy.interpolate import RegularGridInterpolator

from ._grid import MeshDefinition, build_definition_from_bounds
from ._logging import get_default_logger
from ._validation import (
    as_float_array,
    ensure_path_exists,
    validate_matching_shapes,
    validate_sampling_fraction,
)
from .coordinates import CoordinateType
from .interpolation import interpolate_to_mesh
from .io import (
    load_bathymetry_grid,
    read_mesh_definition,
    save_bathymetry_grid,
    write_mesh_definition,
)
from .plotting import plot_bathymetry, plot_bathymetry_3d


@dataclass
class _MeshState:
    definition: MeshDefinition | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    output_path: str = ""


class MeshStructured:
    """Regular rectangular mesh for bathymetry extraction and interpolation.

    Parameters
    ----------
    key:
        Mesh identifier used in configuration files.
    coord_type:
        Coordinate convention for mesh metadata and axis labels.
    name_logger:
        Logger name.
    """

    def __init__(self, key: str, coord_type: CoordinateType | str = CoordinateType.UTM, utm_zone_number: int | None = None, utm_zone_letter: str | None = None, name_logger: str = "Mesh_Structured",) -> None:
        self.logger = get_default_logger(name_logger)
        self.key = key
        self.coord_type = CoordinateType.coerce(coord_type)
        self.utm_zone_number = utm_zone_number
        self.utm_zone_letter = utm_zone_letter
        self._state = _MeshState()

    @property
    def xmin(self) -> float | None:
        """Minimum x coordinate of the mesh origin."""

        return None if self._state.definition is None else self._state.definition.xmin

    @property
    def ymin(self) -> float | None:
        """Minimum y coordinate of the mesh origin."""

        return None if self._state.definition is None else self._state.definition.ymin

    @property
    def dx(self) -> float | None:
        """Mesh spacing along x."""

        return None if self._state.definition is None else self._state.definition.dx

    @property
    def dy(self) -> float | None:
        """Mesh spacing along y."""

        return None if self._state.definition is None else self._state.definition.dy

    @property
    def nx(self) -> int | None:
        """Number of cells along x."""

        return None if self._state.definition is None else self._state.definition.nx

    @property
    def ny(self) -> int | None:
        """Number of cells along y."""

        return None if self._state.definition is None else self._state.definition.ny

    @property
    def lx(self) -> float | None:
        """Mesh extent along x."""

        return None if self._state.definition is None else self._state.definition.lx

    @property
    def ly(self) -> float | None:
        """Mesh extent along y."""

        return None if self._state.definition is None else self._state.definition.ly

    @property
    def x(self) -> np.ndarray | None:
        """Mesh x coordinates."""

        return self._state.x

    @property
    def y(self) -> np.ndarray | None:
        """Mesh y coordinates."""

        return self._state.y

    @property
    def z(self) -> np.ndarray | None:
        """Mesh bathymetry values."""

        return self._state.z

    @property
    def fname_out(self) -> str:
        """Last output file path used for bathymetry IO."""

        return self._state.output_path

    def read_conf(self, file_mesh_ini: str | Path) -> None:
        """Load mesh parameters from an INI file."""

        self._state.definition = read_mesh_definition(
            file_mesh_ini,
            key=self.key,
            coordinate_type=self.coord_type,
        )
        self.logger.info("Loaded mesh configuration from %s for key %s.", file_mesh_ini, self.key)

    def save_conf(self, file_mesh_ini_save: str | Path) -> None:
        """Save the current mesh definition to an INI file."""

        definition = self._require_definition()
        write_mesh_definition(
            file_mesh_ini_save,
            key=self.key,
            coordinate_type=self.coord_type,
            definition=definition,
        )
        self.logger.info("Saved mesh configuration to %s for key %s.", file_mesh_ini_save, self.key)

    def execute(
        self,
        xb: Any,
        yb: Any,
        zb: Any,
        x1: float | None = None,
        x2: float | None = None,
        y1: float | None = None,
        y2: float | None = None,
        dx: float = 100,
        dy: float = 100,
        file_mesh_ini: str | Path | None = None,
        xc: Any | None = None,
        yc: Any | None = None,
        factor_select: float = 1,
        random_seed: int | None = None,
    ) -> None:
        """Compute bathymetry on the structured mesh.

        For curvilinear UTM source coordinates, provide ``utm_zone_number`` and
        ``utm_zone_letter`` to interpolate on the original rectilinear
        longitude/latitude grid without triangulating the projected points.
        """

        xb_array = as_float_array(xb, name="xb")
        yb_array = as_float_array(yb, name="yb")
        zb_array = as_float_array(zb, name="zb")
        validate_matching_shapes(
            names=("xb", "yb", "zb"),
            arrays=(xb_array, yb_array, zb_array),
        )
        validate_sampling_fraction(factor_select)

        self._set_definition(
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            dx=dx,
            dy=dy,
            file_mesh_ini=file_mesh_ini,
        )
        self._build_mesh_coordinates()
        self.logger.info("Structured mesh coordinates generated for key %s.", self.key)

        if factor_select < 1:
            generator = np.random.default_rng(random_seed)
            original_size = len(xb_array)
            sample_size = min(original_size, max(2, int(original_size * factor_select)))
            indices = np.sort(generator.choice(original_size, size=sample_size, replace=False))
            xb_array = xb_array[indices]
            yb_array = yb_array[indices]
            zb_array = zb_array[indices]
            self.logger.info(
                "Selected %s bathymetry rows out of %s using factor_select=%s.",
                sample_size,
                original_size,
                factor_select,
            )

        try:
            x_axis, y_axis = self._rectilinear_axes(xb_array, yb_array)
        except ValueError:
            if self.coord_type is CoordinateType.UTM and (
                self.utm_zone_number is not None or self.utm_zone_letter is not None
            ):
                if self.utm_zone_number is None or self.utm_zone_letter is None:
                    raise ValueError(
                        "utm_zone_number and utm_zone_letter must be provided together."
                    )
                self._state.z = self._interpolate_utm_via_lonlat(
                    xb_array,
                    yb_array,
                    zb_array,
                    zone_number=self.utm_zone_number,
                    zone_letter=self.utm_zone_letter,
                )
            else:
                _, _, self._state.z = interpolate_to_mesh(
                    self._require_x(),
                    self._require_y(),
                    xb_array,
                    yb_array,
                    zb_array,
                )
        else:
            interpolator = RegularGridInterpolator(
                (y_axis, x_axis),
                zb_array,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            self._state.z = interpolator((self._require_y(), self._require_x()))
        self.logger.info("Bathymetry interpolation completed for key %s.", self.key)

        if xc is not None or yc is not None:
            if xc is None or yc is None:
                raise ValueError("Both xc and yc must be provided together when applying a polygon mask.")
            self._apply_polygon_mask(xc=xc, yc=yc)
            self.logger.info("Polygon mask applied to interpolated bathymetry.")

    def get(
        self,
        file_mesh: str | Path,
        x1: float | None = None,
        x2: float | None = None,
        y1: float | None = None,
        y2: float | None = None,
        dx: float = 100,
        dy: float = 100,
        file_mesh_ini: str | Path | None = None,
    ) -> None:
        """Load bathymetry values for an existing structured mesh."""

        self._set_definition(
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            dx=dx,
            dy=dy,
            file_mesh_ini=file_mesh_ini,
        )
        self._build_mesh_coordinates()
        self.load_z(file_mesh)
        self.logger.info("Loaded bathymetry values from %s.", file_mesh)

    def enabled(self) -> bool:
        """Return whether mesh coordinates have been generated."""

        return self._state.x is not None and self._state.y is not None

    def save_z(self, file_save_dat: str | Path) -> None:
        """Save mesh bathymetry to the legacy tab-separated format."""

        save_bathymetry_grid(file_save_dat, self._require_z())
        self._state.output_path = str(file_save_dat)
        self.logger.info("Saved bathymetry grid to %s.", file_save_dat)

    def load_z(self, file_mesh: str | Path) -> None:
        """Load mesh bathymetry from the legacy tab-separated format."""

        ensure_path_exists(file_mesh)
        self._state.z = load_bathymetry_grid(file_mesh)
        self._state.output_path = str(file_mesh)

    def plot(
        self,
        ax: Axes | None = None,
        fname_png: str | Path | None = None,
        _show: bool = True,
        dpi: int = 300,
    ) -> Axes:
        """Plot mesh bathymetry in 2D."""

        return plot_bathymetry(
            self._require_x(),
            self._require_y(),
            self._require_z(),
            self.coord_type,
            ax=ax,
            output_path=fname_png,
            show=_show,
            dpi=dpi,
        )

    def plot_3d(
        self,
        ax: Axes | None = None,
        fname_png: str | Path | None = None,
        _show: bool = True,
        dpi: int = 300,
    ) -> Axes:
        """Plot mesh bathymetry in 3D."""

        return plot_bathymetry_3d(
            self._require_x(),
            self._require_y(),
            self._require_z(),
            self.coord_type,
            ax=ax,
            output_path=fname_png,
            show=_show,
            dpi=dpi,
        )

    def interpolate(self, x: Any, y: Any, var: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate a scalar field onto the current structured mesh."""

        definition = self._require_definition()
        x_axis = np.arange(definition.xmin, definition.xmin + definition.lx, definition.dx, dtype=float)
        y_axis = np.arange(definition.ymin, definition.ymin + definition.ly, definition.dy, dtype=float)
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        result = interpolate_to_mesh(x_mesh, y_mesh, x, y, var)
        self.logger.info("Interpolated external variable onto structured mesh.")
        return result

    def _set_definition(
        self,
        *,
        x1: float | None,
        x2: float | None,
        y1: float | None,
        y2: float | None,
        dx: float,
        dy: float,
        file_mesh_ini: str | Path | None,
    ) -> None:
        if file_mesh_ini is not None and Path(file_mesh_ini).exists():
            self.read_conf(file_mesh_ini)
            return

        if None in {x1, x2, y1, y2}:
            raise ValueError(
                "Mesh bounds x1, x2, y1 and y2 are required when file_mesh_ini is not provided "
                "or does not exist."
            )

        self._state.definition = build_definition_from_bounds(
            x1=float(x1),
            x2=float(x2),
            y1=float(y1),
            y2=float(y2),
            dx=dx,
            dy=dy,
        )
        self.logger.info(
            "Created mesh definition from bounds x=(%s, %s), y=(%s, %s), spacing=(%s, %s).",
            x1,
            x2,
            y1,
            y2,
            dx,
            dy,
        )
        if file_mesh_ini is not None:
            self.save_conf(file_mesh_ini)

    def _build_mesh_coordinates(self) -> None:
        definition = self._require_definition()
        x_mesh, y_mesh = definition.build_coordinates()
        self._state.x = x_mesh
        self._state.y = y_mesh

    def _apply_polygon_mask(self, *, xc: Any, yc: Any) -> None:
        x_contour = as_float_array(xc, name="xc")
        y_contour = as_float_array(yc, name="yc")
        validate_matching_shapes(names=("xc", "yc"), arrays=(x_contour, y_contour))
        contour = MatplotlibPath(np.column_stack((x_contour, y_contour)))
        points = np.column_stack((self._require_x().ravel(), self._require_y().ravel()))
        mask = contour.contains_points(points).reshape(self._require_x().shape)
        self._state.z = np.where(mask, self._require_z(), np.nan)

    def _interpolate_utm_via_lonlat(
        self,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        values: np.ndarray,
        *,
        zone_number: int,
        zone_letter: str,
    ) -> np.ndarray:
        source_lat, source_lon = utm.to_latlon(
            x_coordinates,
            y_coordinates,
            zone_number,
            zone_letter=zone_letter,
        )
        lon_axis, lat_axis = self._rectilinear_axes(source_lon, source_lat)

        target_lat, target_lon = utm.to_latlon(
            self._require_x(),
            self._require_y(),
            zone_number,
            zone_letter=zone_letter,
        )
        interpolator = RegularGridInterpolator(
            (lat_axis, lon_axis),
            values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        return interpolator((target_lat, target_lon))

    @staticmethod
    def _rectilinear_axes(
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the axes from coordinate arrays created by ``meshgrid``."""

        if x_coordinates.ndim != 2 or y_coordinates.ndim != 2:
            raise ValueError("xb and yb must be two-dimensional rectangular mesh coordinates.")

        x_axis = x_coordinates[0, :]
        y_axis = y_coordinates[:, 0]
        if not np.allclose(x_coordinates, x_axis[None, :], equal_nan=False) or not np.allclose(
            y_coordinates,
            y_axis[:, None],
            equal_nan=False,
        ):
            raise ValueError("xb and yb must define a rectilinear mesh created with numpy.meshgrid.")

        for name, axis in (("xb", x_axis), ("yb", y_axis)):
            differences = np.diff(axis)
            if differences.size == 0 or not (
                np.all(differences > 0) or np.all(differences < 0)
            ):
                raise ValueError(f"The {name} axis must be strictly ascending or descending.")

        return x_axis, y_axis

    def _require_definition(self) -> MeshDefinition:
        if self._state.definition is None:
            raise RuntimeError("Mesh definition is not available. Load or create a mesh first.")
        return self._state.definition

    def _require_x(self) -> np.ndarray:
        if self._state.x is None:
            raise RuntimeError("Mesh x coordinates are not available. Run execute() or get() first.")
        return self._state.x

    def _require_y(self) -> np.ndarray:
        if self._state.y is None:
            raise RuntimeError("Mesh y coordinates are not available. Run execute() or get() first.")
        return self._state.y

    def _require_z(self) -> np.ndarray:
        if self._state.z is None:
            raise RuntimeError("Bathymetry values are not available. Run execute() or load_z() first.")
        return self._state.z
