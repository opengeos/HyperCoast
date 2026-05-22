import numpy as np
import pytest
import xarray as xr

from hypercoast.aviris import (
    extract_aviris,
    get_aviris_asset_url,
    get_aviris_bounds,
    read_aviris,
)


def test_get_aviris_asset_url_prefers_requested_asset():
    granule = {
        "links": [
            {
                "href": "https://example.com/AV320250123_RFL_ORT.nc",
                "title": "Download AV320250123_RFL_ORT.nc",
            },
            {
                "href": "https://example.com/AV320250123_RFL_ORT_QL.tif",
                "title": "Download AV320250123_RFL_ORT_QL.tif",
            },
            {
                "href": "s3://bucket/AV320250123_RFL_ORT.nc",
                "title": "S3 AV320250123_RFL_ORT.nc",
            },
        ]
    }

    assert (
        get_aviris_asset_url(granule, asset="RFL_ORT")
        == "https://example.com/AV320250123_RFL_ORT.nc"
    )
    assert (
        get_aviris_asset_url(granule, asset="RFL_ORT", prefer_s3=True)
        == "s3://bucket/AV320250123_RFL_ORT.nc"
    )
    assert (
        get_aviris_asset_url(granule, asset="RFL_ORT_QL")
        == "https://example.com/AV320250123_RFL_ORT_QL.tif"
    )


def test_get_aviris_bounds_from_earthaccess_granule_dict():
    granule = {
        "umm": {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {
                        "GPolygons": [
                            {
                                "Boundary": {
                                    "Points": [
                                        {"Longitude": -118.48, "Latitude": 34.30},
                                        {"Longitude": -118.46, "Latitude": 34.28},
                                        {"Longitude": -118.47, "Latitude": 34.29},
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    }

    assert get_aviris_bounds(granule) == [-118.48, 34.28, -118.46, 34.3]


def test_read_aviris_netcdf_normalizes_av3_av5_layout(tmp_path):
    filepath = tmp_path / "AV320250123_RFL_ORT.nc"
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    ds = xr.Dataset(
        {
            "reflectance": (
                ("wavelength", "northing", "easting"),
                data,
                {"grid_mapping": "transverse_mercator"},
            )
        },
        coords={
            "wavelength": [390.123, 397.456, 404.789],
            "northing": [200.0, 190.0],
            "easting": [10.0, 20.0, 30.0, 40.0],
        },
    )
    ds.to_netcdf(filepath)

    out = read_aviris(str(filepath), chunks=None)

    assert out["reflectance"].dims == ("y", "x", "wavelength")
    assert out.sizes["y"] == 2
    assert out.sizes["x"] == 4
    assert out["wavelength"].values.tolist() == [390.12, 397.46, 404.79]
    assert out.attrs["_FillValue"] == -9999


def test_extract_aviris_uses_grid_mapping_crs_without_dataset_attr():
    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x", "wavelength"),
                np.ones((2, 2, 3), dtype=np.float32),
                {"grid_mapping": "crs"},
            )
        },
        coords={
            "y": [200.0, 100.0],
            "x": [500000.0, 500100.0],
            "wavelength": [400.0, 500.0, 600.0],
            "crs": 0,
        },
    )
    ds["crs"].attrs["spatial_ref"] = (
        'PROJCS["WGS 84 / UTM zone 11N",GEOGCS["WGS 84",'
        'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
        'PARAMETER["central_meridian",-117],PARAMETER["scale_factor",0.9996],'
        'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
        'UNIT["metre",1],AUTHORITY["EPSG","32611"]]'
    )

    da = extract_aviris(ds, lat=0.001, lon=-117.0)

    assert da.dims == ("wavelength",)
    assert da.sizes["wavelength"] == 3


def test_extract_aviris_fails_fast_for_remote_netcdf():
    ds = xr.Dataset(
        {
            "reflectance": (
                ("y", "x", "wavelength"),
                np.ones((2, 2, 3), dtype=np.float32),
            )
        },
        coords={
            "y": [200.0, 100.0],
            "x": [500000.0, 500100.0],
            "wavelength": [400.0, 500.0, 600.0],
        },
        attrs={
            "crs": "EPSG:32611",
            "_source": "https://example.com/AV320250123_RFL_ORT.nc",
            "_source_is_remote": True,
        },
    )

    with pytest.raises(RuntimeError, match="remote AVIRIS NetCDF"):
        extract_aviris(ds, lat=0.001, lon=-117.0)

    da = extract_aviris(ds, lat=0.001, lon=-117.0, allow_remote=True)
    assert da.dims == ("wavelength",)
