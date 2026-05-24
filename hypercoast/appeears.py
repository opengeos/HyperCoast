# SPDX-FileCopyrightText: 2026 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Utilities for submitting and downloading NASA AppEEARS tasks."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

APPEEARS_API_URL = "https://appeears.earthdatacloud.nasa.gov/api/"
EMIT_REFLECTANCE_PRODUCT = "EMIT_L2A_RFL.001"
_WAVELENGTH_RE = re.compile(
    r"Wavelength:\s*([0-9]+(?:\.[0-9]+)?)\s*nm"
    r"(?:,\s*FWHM:\s*([0-9]+(?:\.[0-9]+)?)\s*nm)?"
)
_BAND_RE = re.compile(r"(?:^|[_\-.])(B\d{3})(?:_GW)?(?:$|[_\-.])")


class AppEEARSClient:
    """Client for the NASA AppEEARS API.

    Args:
        api_url: Base URL for the AppEEARS API.
        token: Existing AppEEARS bearer token.
        session: Optional requests-compatible session.
    """

    def __init__(
        self,
        api_url: str = APPEEARS_API_URL,
        token: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> None:
        import requests

        self.api_url = api_url.rstrip("/") + "/"
        self.token = token
        self.session = session or requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        """Return authorization headers for authenticated requests.

        Returns:
            Dictionary containing the bearer token header when a token exists.
        """

        if self.token is None:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate with AppEEARS.

        Args:
            username: NASA Earthdata username. If omitted, environment variables
                or a netrc entry for urs.earthdata.nasa.gov are used.
            password: NASA Earthdata password.
            token: Existing AppEEARS token. When provided, no login request is
                sent.

        Returns:
            AppEEARS login response containing the bearer token.
        """

        if token is not None:
            self.token = token
            return {"token": token}

        if username is None:
            username = os.environ.get("EARTHDATA_USERNAME")
        if password is None:
            password = os.environ.get("EARTHDATA_PASSWORD")

        if username is None or password is None:
            try:
                from netrc import netrc

                auth = netrc().authenticators("urs.earthdata.nasa.gov")
            except (FileNotFoundError, OSError):
                auth = None
            if auth is not None:
                username = username or auth[0]
                password = password or auth[2]

        if username is None or password is None:
            raise ValueError(
                "username and password are required unless EARTHDATA_USERNAME "
                "and EARTHDATA_PASSWORD or a netrc entry are available."
            )

        response = self._request(
            "post", "login", auth=(username, password), auth_header=False
        )
        self.token = response["token"]
        return response

    def products(
        self,
        keyword: Optional[str] = None,
        platform: Optional[str] = None,
        available: Optional[bool] = True,
    ) -> List[Dict[str, Any]]:
        """List AppEEARS products.

        Args:
            keyword: Case-insensitive search text matched against product fields.
            platform: Platform name, such as ``"EMIT"``.
            available: If set, filter by the AppEEARS availability flag.

        Returns:
            List of product metadata dictionaries.
        """

        products = self._request("get", "product", auth_header=False)
        if keyword is not None:
            query = keyword.lower()
            products = [p for p in products if query in json.dumps(p).lower()]
        if platform is not None:
            products = [
                p
                for p in products
                if str(p.get("Platform", "")).lower() == platform.lower()
            ]
        if available is not None:
            products = [p for p in products if p.get("Available") is available]
        return products

    def layers(self, product: str) -> Dict[str, Dict[str, Any]]:
        """List layers for an AppEEARS product.

        Args:
            product: Product and version string, such as ``"EMIT_L2A_RFL.001"``.

        Returns:
            Dictionary keyed by layer name.
        """

        return self._request("get", f"product/{product}", auth_header=False)

    def spectral_layers(
        self,
        product: str = EMIT_REFLECTANCE_PRODUCT,
        wavelengths: Optional[Sequence[float]] = None,
        include_flags: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return spectral layer metadata, optionally nearest to wavelengths.

        Args:
            product: AppEEARS product and version.
            wavelengths: Target wavelengths in nanometers. If omitted, all
                spectral bands are returned.
            include_flags: Whether to include good wavelength flag layers.

        Returns:
            List of layer records containing product, layer, wavelength, fwhm,
            units, description, and data type.
        """

        records = []
        for layer, metadata in self.layers(product).items():
            if not include_flags and layer.endswith("_GW"):
                continue
            pattern = r"B\d{3}(?:_GW)?" if include_flags else r"B\d{3}"
            if not re.fullmatch(pattern, layer):
                continue
            record = _layer_record(product, layer, metadata)
            if record["wavelength"] is not None:
                records.append(record)

        records.sort(key=lambda item: item["wavelength"])
        if wavelengths is None:
            return records

        selected = []
        used_layers = set()
        for wavelength in wavelengths:
            nearest = min(
                records,
                key=lambda item: abs(float(item["wavelength"]) - float(wavelength)),
            )
            if nearest["layer"] not in used_layers:
                selected.append(nearest)
                used_layers.add(nearest["layer"])
        return selected

    def projections(self) -> Dict[str, Any]:
        """List AppEEARS output projections.

        Returns:
            Dictionary of AppEEARS projection metadata.
        """

        return self._request("get", "spatial/proj", auth_header=False)

    def submit_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an AppEEARS task.

        Args:
            task: AppEEARS task payload.

        Returns:
            AppEEARS task submission response.
        """

        self._require_token()
        return self._request("post", "task", json=task)

    def tasks(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        pretty: bool = False,
    ) -> Dict[str, Any]:
        """List submitted AppEEARS tasks for the authenticated user.

        Args:
            limit: Maximum number of tasks to return.
            offset: Result offset for pagination.
            pretty: Whether AppEEARS should pretty-print the JSON response.

        Returns:
            Task list response from AppEEARS.
        """

        self._require_token()
        params = {"pretty": pretty}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._request("get", "task", params=params)

    def task(self, task_id: str) -> Dict[str, Any]:
        """Get an AppEEARS task.

        Args:
            task_id: AppEEARS task identifier.

        Returns:
            Task metadata response.
        """

        self._require_token()
        return self._request("get", f"task/{task_id}")

    def status(self, task_id: str) -> Dict[str, Any]:
        """Get an AppEEARS task status.

        Args:
            task_id: AppEEARS task identifier.

        Returns:
            Task status response.
        """

        self._require_token()
        return self._request("get", f"status/{task_id}")

    def wait_for_task(
        self,
        task_id: str,
        interval: int = 20,
        timeout: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Wait for an AppEEARS task to finish.

        Args:
            task_id: AppEEARS task identifier.
            interval: Polling interval in seconds.
            timeout: Optional timeout in seconds.
            verbose: Whether to print status changes.

        Returns:
            Final task metadata response.
        """

        start = time.time()
        last_status = None
        while True:
            response = self.task(task_id)
            status = response.get("status")
            if verbose and status != last_status:
                print(status)
            if status == "done":
                return response
            if status in {"error", "failed"}:
                raise RuntimeError(f"AppEEARS task {task_id} failed: {response}")
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for AppEEARS task {task_id}.")
            last_status = status
            time.sleep(interval)

    def bundle(self, task_id: str) -> Dict[str, Any]:
        """Get bundle metadata for a completed AppEEARS task.

        Args:
            task_id: AppEEARS task identifier.

        Returns:
            Bundle metadata response.
        """

        self._require_token()
        return self._request("get", f"bundle/{task_id}")

    def download_bundle(
        self,
        task_id: str,
        out_dir: Union[str, os.PathLike[str]] = ".",
        file_ids: Optional[Iterable[str]] = None,
        file_types: Optional[Iterable[str]] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """Download files from an AppEEARS task bundle.

        Args:
            task_id: AppEEARS task identifier.
            out_dir: Output directory.
            file_ids: Optional subset of AppEEARS file identifiers to download.
            file_types: Optional file type filter, such as ``["tif", "csv"]``.
            overwrite: Whether to overwrite existing files.

        Returns:
            List of downloaded file paths.
        """

        bundle = self.bundle(task_id)
        selected_file_ids = set(file_ids) if file_ids is not None else None
        selected_file_types = set(file_types) if file_types is not None else None
        paths = []
        for item in bundle.get("files", []):
            file_id = item["file_id"]
            if selected_file_ids is not None and file_id not in selected_file_ids:
                continue
            if (
                selected_file_types is not None
                and item.get("file_type") not in selected_file_types
            ):
                continue
            path = self.download_file(
                task_id=task_id,
                file_id=file_id,
                file_name=item["file_name"],
                out_dir=out_dir,
                overwrite=overwrite,
            )
            paths.append(path)
        return paths

    def download_file(
        self,
        task_id: str,
        file_id: str,
        file_name: str,
        out_dir: Union[str, os.PathLike[str]] = ".",
        overwrite: bool = False,
        chunk_size: int = 8192,
    ) -> str:
        """Download one AppEEARS bundle file.

        Args:
            task_id: AppEEARS task identifier.
            file_id: Bundle file identifier.
            file_name: Bundle file name.
            out_dir: Output directory.
            overwrite: Whether to overwrite an existing file.
            chunk_size: Response chunk size in bytes.

        Returns:
            Path to the downloaded file.
        """

        self._require_token()
        out_path = _bundle_output_path(out_dir, file_name)
        if out_path.exists() and not overwrite:
            return str(out_path)

        response = self._raw_request("get", f"bundle/{task_id}/{file_id}", stream=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as dst:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    dst.write(chunk)
        return str(out_path)

    def s3credentials(self) -> Dict[str, Any]:
        """Request temporary S3 credentials from AppEEARS.

        Returns:
            Temporary credential response from AppEEARS.
        """

        self._require_token()
        return self._request("get", "s3credentials")

    def _request(
        self,
        method: str,
        path: str,
        auth_header: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Send a JSON request to AppEEARS.

        Args:
            method: HTTP method name.
            path: API path relative to the base URL.
            auth_header: Whether to include the bearer token header.
            **kwargs: Additional request keyword arguments.

        Returns:
            Decoded JSON response.
        """

        response = self._raw_request(method, path, auth_header=auth_header, **kwargs)
        try:
            return response.json()
        except ValueError:
            return response.text

    def _raw_request(
        self,
        method: str,
        path: str,
        auth_header: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Send a raw request to AppEEARS.

        Args:
            method: HTTP method name.
            path: API path relative to the base URL.
            auth_header: Whether to include the bearer token header.
            **kwargs: Additional request keyword arguments.

        Returns:
            Requests response object.
        """

        headers = kwargs.pop("headers", {})
        if auth_header:
            headers = {**self.headers, **headers}
        kwargs.setdefault("timeout", 120)
        response = self.session.request(
            method.upper(), self.api_url + path.lstrip("/"), headers=headers, **kwargs
        )
        response.raise_for_status()
        return response

    def _require_token(self) -> None:
        """Raise an error when an authenticated request lacks a token."""

        if self.token is None:
            raise ValueError("Call login() or provide a token before this request.")


def appeears_login(
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    api_url: str = APPEEARS_API_URL,
) -> AppEEARSClient:
    """Create and authenticate an AppEEARS client.

    Args:
        username: NASA Earthdata username.
        password: NASA Earthdata password.
        token: Existing AppEEARS token.
        api_url: Base URL for the AppEEARS API.

    Returns:
        Authenticated AppEEARS client.
    """

    client = AppEEARSClient(api_url=api_url)
    client.login(username=username, password=password, token=token)
    return client


def appeears_products(
    keyword: Optional[str] = None,
    platform: Optional[str] = None,
    available: Optional[bool] = True,
    client: Optional[AppEEARSClient] = None,
) -> List[Dict[str, Any]]:
    """List AppEEARS products.

    Args:
        keyword: Case-insensitive search text matched against product fields.
        platform: Platform name, such as ``"EMIT"``.
        available: If set, filter by the AppEEARS availability flag.
        client: Optional AppEEARS client.

    Returns:
        List of product metadata dictionaries.
    """

    client = client or AppEEARSClient()
    return client.products(keyword=keyword, platform=platform, available=available)


def appeears_layers(
    product: str,
    client: Optional[AppEEARSClient] = None,
) -> Dict[str, Dict[str, Any]]:
    """List AppEEARS layers for a product.

    Args:
        product: Product and version string.
        client: Optional AppEEARS client.

    Returns:
        Dictionary keyed by layer name.
    """

    client = client or AppEEARSClient()
    return client.layers(product)


def appeears_emit_layers(
    wavelengths: Optional[Sequence[float]] = None,
    product: str = EMIT_REFLECTANCE_PRODUCT,
    include_flags: bool = False,
    return_metadata: bool = False,
    client: Optional[AppEEARSClient] = None,
) -> List[Dict[str, Any]]:
    """Select EMIT AppEEARS layers by wavelength.

    Args:
        wavelengths: Target wavelengths in nanometers. If omitted, all spectral
            EMIT bands are returned.
        product: EMIT product and version.
        include_flags: Whether to include good wavelength flag layers.
        return_metadata: Whether to return wavelength metadata. If False,
            task-ready ``{"product": product, "layer": layer}`` dictionaries
            are returned.
        client: Optional AppEEARS client.

    Returns:
        Selected EMIT layer records or task-ready layer dictionaries.
    """

    client = client or AppEEARSClient()
    layers = client.spectral_layers(
        product=product, wavelengths=wavelengths, include_flags=include_flags
    )
    if return_metadata:
        return layers
    return [{"product": item["product"], "layer": item["layer"]} for item in layers]


def appeears_point_task(
    task_name: str,
    coordinates: Union[Sequence[float], Sequence[Dict[str, Any]]],
    layers: Sequence[Dict[str, str]],
    start_date: str,
    end_date: str,
    recurring: bool = False,
    year_range: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build an AppEEARS point task payload.

    Args:
        task_name: AppEEARS task name.
        coordinates: One ``(longitude, latitude)`` pair or AppEEARS coordinate
            dictionaries containing longitude and latitude.
        layers: Task-ready AppEEARS layers.
        start_date: Start date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        end_date: End date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        recurring: Whether the task uses recurring dates.
        year_range: Two-year range for recurring tasks.

    Returns:
        AppEEARS point task payload.
    """

    params = {
        "dates": [
            _date_range(
                start_date=start_date,
                end_date=end_date,
                recurring=recurring,
                year_range=year_range,
            )
        ],
        "layers": list(layers),
        "coordinates": _normalize_coordinates(coordinates),
    }
    return {"task_type": "point", "task_name": task_name, "params": params}


def appeears_area_task(
    task_name: str,
    geometry: Any,
    layers: Sequence[Dict[str, str]],
    start_date: str,
    end_date: str,
    output_format: str = "geotiff",
    projection: str = "geographic",
    recurring: bool = False,
    year_range: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build an AppEEARS area task payload.

    Args:
        task_name: AppEEARS task name.
        geometry: GeoJSON geometry, feature, feature collection, GeoDataFrame,
            vector file path, or bounding box ``[xmin, ymin, xmax, ymax]``.
        layers: Task-ready AppEEARS layers.
        start_date: Start date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        end_date: End date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        output_format: AppEEARS output format, such as ``"geotiff"`` or
            ``"netcdf4"``.
        projection: AppEEARS output projection name.
        recurring: Whether the task uses recurring dates.
        year_range: Two-year range for recurring tasks.

    Returns:
        AppEEARS area task payload.
    """

    params = {
        "dates": [
            _date_range(
                start_date=start_date,
                end_date=end_date,
                recurring=recurring,
                year_range=year_range,
            )
        ],
        "layers": list(layers),
        "output": {"format": {"type": output_format}, "projection": projection},
        "geo": _normalize_geojson(geometry),
    }
    return {"task_type": "area", "task_name": task_name, "params": params}


def appeears_submit_task(
    task: Dict[str, Any],
    client: Optional[AppEEARSClient] = None,
) -> Dict[str, Any]:
    """Submit an AppEEARS task.

    Args:
        task: AppEEARS task payload.
        client: Authenticated AppEEARS client.

    Returns:
        AppEEARS task submission response.
    """

    if client is None:
        client = appeears_login()
    return client.submit_task(task)


def appeears_wait(
    task_id: str,
    client: Optional[AppEEARSClient] = None,
    interval: int = 20,
    timeout: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Wait for an AppEEARS task to finish.

    Args:
        task_id: AppEEARS task identifier.
        client: Authenticated AppEEARS client.
        interval: Polling interval in seconds.
        timeout: Optional timeout in seconds.
        verbose: Whether to print status changes.

    Returns:
        Final task metadata response.
    """

    if client is None:
        client = appeears_login()
    return client.wait_for_task(
        task_id=task_id, interval=interval, timeout=timeout, verbose=verbose
    )


def appeears_download(
    task_id: str,
    out_dir: Union[str, os.PathLike[str]] = ".",
    client: Optional[AppEEARSClient] = None,
    file_types: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> List[str]:
    """Download an AppEEARS task bundle.

    Args:
        task_id: AppEEARS task identifier.
        out_dir: Output directory.
        client: Authenticated AppEEARS client.
        file_types: Optional file type filter, such as ``["tif"]``.
        overwrite: Whether to overwrite existing files.

    Returns:
        List of downloaded file paths.
    """

    if client is None:
        client = appeears_login()
    return client.download_bundle(
        task_id=task_id,
        out_dir=out_dir,
        file_types=file_types,
        overwrite=overwrite,
    )


def read_appeears(
    files: Union[str, os.PathLike[str], Sequence[Union[str, os.PathLike[str]]]],
    layers: Optional[Union[Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]]] = None,
    product: str = EMIT_REFLECTANCE_PRODUCT,
    variable: str = "reflectance",
    client: Optional[AppEEARSClient] = None,
    **kwargs: Any,
) -> Any:
    """Read AppEEARS single-band GeoTIFFs as an xarray hyperspectral cube.

    Args:
        files: GeoTIFF path, directory, or sequence of paths.
        layers: Optional AppEEARS layer metadata from ``appeears_layers`` or
            ``appeears_emit_layers(..., return_metadata=True)``.
        product: AppEEARS product used to infer wavelength metadata when layers
            are omitted.
        variable: Output data variable name.
        client: Optional AppEEARS client used to fetch layer metadata.
        **kwargs: Additional keyword arguments passed to
            ``rioxarray.open_rasterio``.

    Returns:
        xarray.Dataset containing the stacked raster data.
    """

    import rioxarray
    import xarray as xr

    paths = _coerce_paths(files)
    if layers is None:
        client = client or AppEEARSClient()
        layers = client.layers(product)
    metadata = _metadata_by_layer(product, layers)

    arrays = []
    wavelengths = []
    band_names = []
    for path in paths:
        layer_name = _layer_from_filename(path)
        if layer_name is None:
            continue
        layer_info = metadata.get(layer_name)
        wavelength = None if layer_info is None else layer_info.get("wavelength")
        if wavelength is None:
            continue
        data = rioxarray.open_rasterio(path, masked=True, **kwargs)
        if "band" in data.dims and data.sizes.get("band") == 1:
            data = data.squeeze("band", drop=True)
        arrays.append(data)
        wavelengths.append(float(wavelength))
        band_names.append(layer_name)

    if not arrays:
        raise ValueError("No AppEEARS spectral GeoTIFF files could be read.")

    order = sorted(range(len(wavelengths)), key=lambda index: wavelengths[index])
    arrays = [arrays[index] for index in order]
    wavelengths = [wavelengths[index] for index in order]
    band_names = [band_names[index] for index in order]
    data = xr.concat(
        arrays,
        dim=xr.DataArray(wavelengths, dims="wavelength", name="wavelength"),
    )
    dataset = data.to_dataset(name=variable)
    dataset[variable].attrs["band_names"] = band_names
    dataset.attrs["source"] = "NASA AppEEARS"
    dataset.attrs["product"] = product
    return dataset


def _layer_record(
    product: str,
    layer: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a normalized AppEEARS layer metadata record.

    Args:
        product: AppEEARS product and version.
        layer: AppEEARS layer name.
        metadata: Layer metadata response.

    Returns:
        Normalized layer metadata record.
    """

    wavelength, fwhm = _parse_wavelength(metadata.get("Description", ""))
    return {
        "product": product,
        "layer": layer,
        "wavelength": wavelength,
        "fwhm": fwhm,
        "units": metadata.get("Units"),
        "description": metadata.get("Description"),
        "data_type": metadata.get("DataType"),
    }


def _parse_wavelength(description: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse wavelength and FWHM values from AppEEARS layer descriptions.

    Args:
        description: AppEEARS layer description.

    Returns:
        Tuple containing wavelength and FWHM in nanometers.
    """

    match = _WAVELENGTH_RE.search(description or "")
    if match is None:
        return None, None
    wavelength = float(match.group(1))
    fwhm = float(match.group(2)) if match.group(2) is not None else None
    return wavelength, fwhm


def _date_range(
    start_date: str,
    end_date: str,
    recurring: bool = False,
    year_range: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build an AppEEARS date range dictionary.

    Args:
        start_date: Start date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        end_date: End date as ``YYYY-MM-DD`` or ``MM-DD-YYYY``.
        recurring: Whether the task uses recurring dates.
        year_range: Two-year range for recurring tasks.

    Returns:
        AppEEARS date range dictionary.
    """

    date = {
        "startDate": _format_appeears_date(start_date),
        "endDate": _format_appeears_date(end_date),
    }
    if recurring:
        date["recurring"] = True
        if year_range is not None:
            date["yearRange"] = list(year_range)
    return date


def _format_appeears_date(value: str) -> str:
    """Format a date for AppEEARS.

    Args:
        value: Date string in ``YYYY-MM-DD`` or ``MM-DD-YYYY`` format.

    Returns:
        Date string in ``MM-DD-YYYY`` format.
    """

    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match is not None:
        year, month, day = match.groups()
        return f"{month}-{day}-{year}"
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", value):
        return value
    raise ValueError("Dates must use YYYY-MM-DD or MM-DD-YYYY format.")


def _normalize_coordinates(
    coordinates: Union[Sequence[float], Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Normalize point coordinates for AppEEARS.

    Args:
        coordinates: One ``(longitude, latitude)`` pair or a sequence of
            AppEEARS coordinate dictionaries.

    Returns:
        List of AppEEARS coordinate dictionaries.
    """

    if len(coordinates) == 2 and all(
        isinstance(value, (int, float)) for value in coordinates
    ):
        lon, lat = coordinates
        return [{"longitude": lon, "latitude": lat, "id": "point-1"}]

    normalized = []
    for index, coord in enumerate(coordinates):
        item = dict(coord)
        if "id" not in item:
            item["id"] = f"point-{index + 1}"
        normalized.append(item)
    return normalized


def _normalize_geojson(geometry: Any) -> Dict[str, Any]:
    """Normalize area geometry for AppEEARS.

    Args:
        geometry: GeoJSON, GeoDataFrame, vector path, or bounding box.

    Returns:
        GeoJSON feature collection, feature, or geometry dictionary.
    """

    if isinstance(geometry, (str, os.PathLike)):
        import geopandas as gpd

        return json.loads(gpd.read_file(geometry).to_json())

    if (
        isinstance(geometry, (list, tuple))
        and len(geometry) == 4
        and all(isinstance(v, (int, float)) for v in geometry)
    ):
        xmin, ymin, xmax, ymax = geometry
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax],
                                [xmin, ymin],
                            ]
                        ],
                    },
                }
            ],
        }

    if hasattr(geometry, "to_json"):
        return json.loads(geometry.to_json())

    if hasattr(geometry, "__geo_interface__"):
        return geometry.__geo_interface__

    if isinstance(geometry, dict):
        return geometry

    raise TypeError(
        "geometry must be GeoJSON, a GeoDataFrame, a vector file path, "
        "or a bounding box."
    )


def _bundle_output_path(
    out_dir: Union[str, os.PathLike[str]],
    file_name: str,
) -> Path:
    """Build a local bundle output path.

    Args:
        out_dir: Output directory.
        file_name: AppEEARS bundle file name.

    Returns:
        Local output path.
    """

    file_path = Path(file_name)
    if file_path.is_absolute() or ".." in file_path.parts:
        file_path = Path(file_path.name)
    parts = file_path.parts
    if len(parts) > 1 and file_path.suffix.lower() == ".tif":
        file_path = Path(parts[-1])
    return Path(out_dir) / file_path


def _coerce_paths(
    files: Union[str, os.PathLike[str], Sequence[Union[str, os.PathLike[str]]]],
) -> List[str]:
    """Coerce file input to a sorted list of GeoTIFF paths.

    Args:
        files: Path, directory, or sequence of paths.

    Returns:
        Sorted list of GeoTIFF paths.
    """

    if isinstance(files, (str, os.PathLike)):
        path = Path(files)
        if path.is_dir():
            return sorted(str(item) for item in path.rglob("*.tif"))
        return [str(path)]
    return sorted(str(Path(item)) for item in files)


def _metadata_by_layer(
    product: str,
    layers: Union[Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Index AppEEARS layer metadata by layer name.

    Args:
        product: AppEEARS product and version.
        layers: AppEEARS layer metadata.

    Returns:
        Dictionary keyed by layer name.
    """

    if isinstance(layers, dict):
        return {
            layer: _layer_record(product, layer, metadata)
            for layer, metadata in layers.items()
        }
    return {item["layer"]: dict(item) for item in layers}


def _layer_from_filename(path: Union[str, os.PathLike[str]]) -> Optional[str]:
    """Extract an AppEEARS layer name from a file path.

    Args:
        path: AppEEARS output file path.

    Returns:
        Layer name when a band token is present.
    """

    matches = _BAND_RE.findall(Path(path).name)
    if not matches:
        return None
    return matches[-1]
