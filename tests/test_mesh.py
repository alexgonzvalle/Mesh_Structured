from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import utm

from Mesh_Structured import CoordinateType, MeshStructured


def test_execute_generates_expected_mesh(rectangular_plane_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    xb, yb, zb = rectangular_plane_data
    mesh = MeshStructured("main", coord_type=CoordinateType.UTM)

    mesh.execute(xb, yb, zb, x1=0.0, x2=1.0, y1=0.0, y2=1.0, dx=0.5, dy=0.5)

    expected_x = np.array([[0.0, 0.5], [0.0, 0.5]])
    expected_y = np.array([[0.5, 0.5], [0.0, 0.0]])
    expected_z = expected_x + 2.0 * expected_y

    npt.assert_allclose(mesh.x, expected_x)
    npt.assert_allclose(mesh.y, expected_y)
    npt.assert_allclose(mesh.z, expected_z)
    assert mesh.enabled() is True


def test_execute_with_polygon_mask_replaces_values_outside_polygon(
    rectangular_plane_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    xb, yb, zb = rectangular_plane_data
    mesh = MeshStructured("main")

    mesh.execute(
        xb,
        yb,
        zb,
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        dx=0.5,
        dy=0.5,
        xc=np.array([-0.1, 0.25, 0.25, -0.1]),
        yc=np.array([-0.1, -0.1, 0.25, 0.25]),
    )

    assert np.isnan(mesh.z[0, 0])
    assert np.isfinite(mesh.z[1, 0])


def test_configuration_round_trip(tmp_path) -> None:
    mesh = MeshStructured("main", coord_type="LONLAT")
    config_path = tmp_path / "mesh.ini"
    source_axis = np.array([0.0, 1.0])
    source_x, source_y = np.meshgrid(source_axis, source_axis)

    mesh.execute(
        xb=source_x,
        yb=source_y,
        zb=source_x + source_y,
        x1=1.0,
        x2=3.0,
        y1=4.0,
        y2=6.0,
        dx=1.0,
        dy=1.0,
        file_mesh_ini=config_path,
    )

    reloaded_mesh = MeshStructured("main", coord_type="LONLAT")
    reloaded_mesh.read_conf(config_path)

    assert reloaded_mesh.xmin == 1.0
    assert reloaded_mesh.ymin == 4.0
    assert reloaded_mesh.dx == 1.0
    assert reloaded_mesh.dy == 1.0
    assert reloaded_mesh.nx == 2
    assert reloaded_mesh.ny == 2


def test_save_and_load_bathymetry_round_trip(tmp_path) -> None:
    mesh = MeshStructured("main")
    source_axis = np.array([0.0, 1.0])
    source_x, source_y = np.meshgrid(source_axis, source_axis)
    mesh.execute(
        xb=source_x,
        yb=source_y,
        zb=-(1.0 + source_x + 2.0 * source_y),
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        dx=0.5,
        dy=0.5,
    )
    output_path = tmp_path / "bathymetry.dat"

    mesh.save_z(output_path)

    loaded_mesh = MeshStructured("main")
    loaded_mesh.get(
        output_path,
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        dx=0.5,
        dy=0.5,
    )
    npt.assert_allclose(loaded_mesh.z, mesh.z)


def test_execute_requires_bounds_when_configuration_is_missing(rectangular_plane_data) -> None:
    xb, yb, zb = rectangular_plane_data
    mesh = MeshStructured("main")

    with pytest.raises(ValueError, match="Mesh bounds"):
        mesh.execute(xb, yb, zb)


def test_execute_validates_sampling_fraction(rectangular_plane_data) -> None:
    xb, yb, zb = rectangular_plane_data
    mesh = MeshStructured("main")

    with pytest.raises(ValueError, match="factor_select"):
        mesh.execute(xb, yb, zb, x1=0.0, x2=1.0, y1=0.0, y2=1.0, factor_select=1.2)


def test_execute_accepts_descending_source_axes() -> None:
    x_axis = np.array([1.0, 0.5, 0.0])
    y_axis = np.array([1.0, 0.5, 0.0])
    xb, yb = np.meshgrid(x_axis, y_axis)
    mesh = MeshStructured("main")

    mesh.execute(
        xb,
        yb,
        xb + 2.0 * yb,
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        dx=0.5,
        dy=0.5,
    )

    npt.assert_allclose(mesh.z, mesh.x + 2.0 * mesh.y)


def test_execute_accepts_curvilinear_coordinates() -> None:
    mesh = MeshStructured("main")
    xb = np.array([[0.0, 1.0], [0.1, 1.1]])
    yb = np.array([[0.0, 0.0], [1.0, 1.0]])

    mesh.execute(
        xb=xb,
        yb=yb,
        zb=xb + 2.0 * yb,
        x1=0.1,
        x2=1.1,
        y1=0.0,
        y2=1.0,
        dx=0.5,
        dy=0.5,
    )

    npt.assert_allclose(mesh.z, mesh.x + 2.0 * mesh.y)


def test_execute_interpolates_curvilinear_utm_grid_via_lonlat() -> None:
    lon_axis = np.array([-3.01, -3.0, -2.99])
    lat_axis = np.array([43.0, 43.01, 43.02])
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    x_utm, y_utm, _, _ = utm.from_latlon(
        lat,
        lon,
        force_zone_number=30,
        force_zone_letter="N",
    )
    mesh = MeshStructured("main", coord_type=CoordinateType.UTM)

    mesh.execute(
        x_utm,
        y_utm,
        lon + 2.0 * lat,
        x1=float(np.min(x_utm)) + 100.0,
        x2=float(np.max(x_utm)) - 100.0,
        y1=float(np.min(y_utm)) + 100.0,
        y2=float(np.max(y_utm)) - 100.0,
        dx=500.0,
        dy=500.0,
        utm_zone_number=30,
        utm_zone_letter="N",
    )

    target_lat, target_lon = utm.to_latlon(mesh.x, mesh.y, 30, zone_letter="N")
    npt.assert_allclose(mesh.z, target_lon + 2.0 * target_lat, rtol=0.0, atol=3e-6)


def test_interpolate_returns_nan_outside_convex_hull() -> None:
    mesh = MeshStructured("main")
    source_axis = np.array([0.0, 1.0])
    source_x, source_y = np.meshgrid(source_axis, source_axis)
    mesh.execute(
        xb=source_x,
        yb=source_y,
        zb=source_x + source_y,
        x1=0.0,
        x2=2.0,
        y1=0.0,
        y2=2.0,
        dx=1.0,
        dy=1.0,
    )

    x_mesh, y_mesh, values = mesh.interpolate(
        x=np.array([0.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0]),
        var=np.array([1.0, 2.0, 3.0]),
    )

    assert x_mesh.shape == values.shape
    assert y_mesh.shape == values.shape
    assert np.isnan(values[-1, -1])


def test_plot_does_not_mutate_bathymetry(rectangular_plane_data) -> None:
    xb, yb, zb = rectangular_plane_data
    mesh = MeshStructured("main")
    mesh.execute(xb, yb, zb, x1=0.0, x2=1.0, y1=0.0, y2=1.0, dx=0.5, dy=0.5)
    before = mesh.z.copy()

    mesh.plot(_show=False)

    npt.assert_allclose(mesh.z, before)
