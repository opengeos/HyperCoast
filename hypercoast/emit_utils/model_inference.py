"""Model inference and visualization utilities for EMIT data.

This module provides functions for running model inference on EMIT hyperspectral
data, visualizing results with RGB backgrounds, and converting predictions to
GeoTIFF format.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import rasterio

try:
    import torch
    from netCDF4 import Dataset
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    pass


def infer_and_visualize_single_model_Robust(
    model: torch.nn.Module,
    test_loader: DataLoader,
    Rrs: np.ndarray,
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    save_folder: str,
    rgb_nc_file: str,
    structure_name: str,
    TSS_scalers_dict: Dict[str, Any],
    vmin: float = 0,
    vmax: float = 50,
    exposure_coefficient: float = 5.0,
) -> np.ndarray:
    """Run model inference and visualize results with RGB background using robust scaling.

    This function performs model inference on preprocessed EMIT data, applies
    inverse transformations using robust scalers, creates a visualization overlaid
    on an RGB composite, and saves both array and image outputs.

    Args:
        model: Trained PyTorch model for inference.
        test_loader: DataLoader containing preprocessed test data.
        Rrs: Array of remote sensing reflectance data with shape (H, W, B).
        mask: Boolean mask indicating valid pixels with shape (H, W).
        latitude: Array of latitude values.
        longitude: Array of longitude values.
        save_folder: Directory path to save outputs.
        rgb_nc_file: Path to NetCDF file containing RGB bands for visualization.
        structure_name: Name for output files.
        TSS_scalers_dict: Dictionary containing 'log' and 'robust' scalers for
            inverse transformation.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        exposure_coefficient: Coefficient for adjusting RGB brightness.

    Returns:
        final_output: Array of shape (N, 3) containing [lat, lon, value] for each pixel.
    """
    device = next(model.parameters()).device
    predictions_all = []

    # === Model inference ===
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)

            # === Inverse transform using scalers ===
            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(output_dict["pred_y"].cpu().numpy(), dtype=torch.float32)
            )
            predictions_real = (
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )
            predictions_all.append(predictions_real)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # Fill predictions into 2D array according to mask
    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    # Save as [lat, lon, value]
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))
    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    # === Construct RGB image from EMIT L2R .nc ===
    with Dataset(rgb_nc_file) as ds:
        # Latitude
        if "lat" in ds.variables:
            lat_var = ds.variables["lat"][:]
        elif "latitude" in ds.variables:
            lat_var = ds.variables["latitude"][:]
        else:
            raise KeyError("Latitude variable not found")

        # Longitude
        if "lon" in ds.variables:
            lon_var = ds.variables["lon"][:]
        elif "longitude" in ds.variables:
            lon_var = ds.variables["longitude"][:]
        else:
            raise KeyError("Longitude variable not found")

        # rhos band list
        band_list = []
        for name in ds.variables:
            m = re.match(r"^rhos_(\d+(?:\.\d+)?)$", name)
            if m:
                wl = float(m.group(1))
                band_list.append((wl, name))
        if not band_list:
            raise ValueError("No rhos_* bands found")

        # Select nearest RGB bands
        targets = {"R": 664.0, "G": 559.0, "B": 492.0}

        def pick_nearest(target_nm):
            return min(band_list, key=lambda x: abs(x[0] - target_nm))[1]

        var_R = pick_nearest(targets["R"])
        var_G = pick_nearest(targets["G"])
        var_B = pick_nearest(targets["B"])

        R = ds.variables[var_R][:]
        G = ds.variables[var_G][:]
        B = ds.variables[var_B][:]

        if isinstance(R, np.ma.MaskedArray):
            R = R.filled(np.nan)
        if isinstance(G, np.ma.MaskedArray):
            G = G.filled(np.nan)
        if isinstance(B, np.ma.MaskedArray):
            B = B.filled(np.nan)

    # Lat/lon grid
    if lat_var.ndim == 1 and lon_var.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat_var, lon_var, indexing="ij")
    else:
        lat2d, lon2d = lat_var, lon_var

    H, W = R.shape
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    R_flat, G_flat, B_flat = R.reshape(-1), G.reshape(-1), B.reshape(-1)

    lat_top, lat_bot = np.nanmax(lat2d), np.nanmin(lat2d)
    lon_min, lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
    grid_lat = np.linspace(lat_top, lat_bot, H)
    grid_lon = np.linspace(lon_min, lon_max, W)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    R_interp = griddata(
        (lon_flat, lat_flat), R_flat, (grid_lon, grid_lat), method="linear"
    )
    G_interp = griddata(
        (lon_flat, lat_flat), G_flat, (grid_lon, grid_lat), method="linear"
    )
    B_interp = griddata(
        (lon_flat, lat_flat), B_flat, (grid_lon, grid_lat), method="linear"
    )

    rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)
    rgb_max = np.nanmax(rgb_image)
    if not np.isfinite(rgb_max) or rgb_max == 0:
        rgb_max = 1.0
    rgb_image = np.clip((rgb_image / rgb_max) * exposure_coefficient, 0, 1)
    extent_raw = [lon_min, lon_max, lat_bot, lat_top]

    # Interpolate predictions to same grid
    interp_output = griddata(
        (final_output[:, 1], final_output[:, 0]),  # lon, lat
        final_output[:, 2],
        (grid_lon, grid_lat),
        method="linear",
    )
    interp_output = np.ma.masked_invalid(interp_output)

    # Plot and save PNG
    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image, extent=extent_raw, origin="upper")
    im = plt.imshow(
        interp_output,
        extent=extent_raw,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label('(mg m$^{-3}$)', fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {npy_path} (for npy_to_tif)")

    # Return numpy array for direct use
    return final_output


def infer_and_visualize_single_model_minmax(
    model: torch.nn.Module,
    test_loader: DataLoader,
    Rrs: np.ndarray,
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    save_folder: str,
    rgb_nc_file: str,
    structure_name: str,
    vmin: float = 0,
    vmax: float = 50,
    log_offset: float = 0.01,
    exposure_coefficient: float = 5.0,
) -> np.ndarray:
    """Run model inference and visualize results with RGB background using MinMax scaling.

    This function performs model inference on preprocessed EMIT data, applies
    inverse log transformation, creates a visualization overlaid on an RGB
    composite, and saves both array and image outputs.

    Args:
        model: Trained PyTorch model for inference.
        test_loader: DataLoader containing preprocessed test data.
        Rrs: Array of remote sensing reflectance data with shape (H, W, B).
        mask: Boolean mask indicating valid pixels with shape (H, W).
        latitude: Array of latitude values.
        longitude: Array of longitude values.
        save_folder: Directory path to save outputs.
        rgb_nc_file: Path to NetCDF file containing RGB bands for visualization.
        structure_name: Name for output files.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.
        log_offset: Offset used in log transformation during preprocessing.
        exposure_coefficient: Coefficient for adjusting RGB brightness.

    Returns:
        final_output: Array of shape (N, 3) containing [lat, lon, value] for each pixel.
    """
    device = next(model.parameters()).device
    predictions_all = []

    # === Model inference ===
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]
            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    # Fill predictions into 2D array according to mask
    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    # Flatten lat/lon and combine with predictions
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))
    if np.ma.isMaskedArray(final_output):
        final_output = final_output.filled(np.nan)

    # Save .npy file (lat, lon, value)
    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    # === Read RGB bands from .nc file ===
    with Dataset(rgb_nc_file) as ds:
        if "lat" in ds.variables:
            lat_var = ds.variables["lat"][:]
        elif "latitude" in ds.variables:
            lat_var = ds.variables["latitude"][:]
        else:
            raise KeyError("Latitude variable not found (lat/latitude)")

        if "lon" in ds.variables:
            lon_var = ds.variables["lon"][:]
        elif "longitude" in ds.variables:
            lon_var = ds.variables["longitude"][:]
        else:
            raise KeyError("Longitude variable not found (lon/longitude)")

        band_list = []
        for name in ds.variables.keys():
            m = re.match(r"^rhos_(\d+(?:\.\d+)?)$", name)
            if m:
                wl = float(m.group(1))
                band_list.append((wl, name))
        if not band_list:
            raise ValueError("No rhos_* bands found in file")

        targets = {"R": 664.0, "G": 559.0, "B": 492.0}

        def pick_nearest(target_nm):
            idx = int(np.argmin([abs(w - target_nm) for w, _ in band_list]))
            wl_sel, name_sel = band_list[idx]
            return wl_sel, name_sel

        wl_R, var_R = pick_nearest(targets["R"])
        wl_G, var_G = pick_nearest(targets["G"])
        wl_B, var_B = pick_nearest(targets["B"])

        print(
            f"RGB band selection: R→{var_R} (Δ{wl_R - targets['R']:+.1f}nm), "
            f"G→{var_G} (Δ{wl_G - targets['G']:+.1f}nm), "
            f"B→{var_B} (Δ{wl_B - targets['B']:+.1f}nm)"
        )

        R = ds.variables[var_R][:]
        G = ds.variables[var_G][:]
        B = ds.variables[var_B][:]
        if isinstance(R, np.ma.MaskedArray):
            R = R.filled(np.nan)
        if isinstance(G, np.ma.MaskedArray):
            G = G.filled(np.nan)
        if isinstance(B, np.ma.MaskedArray):
            B = B.filled(np.nan)

    if lat_var.ndim == 1 and lon_var.ndim == 1:
        lat2d, lon2d = np.meshgrid(
            np.asarray(lat_var), np.asarray(lon_var), indexing="ij"
        )
    else:
        lat2d, lon2d = np.asarray(lat_var), np.asarray(lon_var)

    H, W = R.shape
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    R_flat, G_flat, B_flat = R.reshape(-1), G.reshape(-1), B.reshape(-1)

    lat_top, lat_bot = np.nanmax(lat2d), np.nanmin(lat2d)
    lon_min, lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
    grid_lat = np.linspace(lat_top, lat_bot, H)
    grid_lon = np.linspace(lon_min, lon_max, W)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    R_interp = griddata(
        (lon_flat, lat_flat), R_flat, (grid_lon, grid_lat), method="linear"
    )
    G_interp = griddata(
        (lon_flat, lat_flat), G_flat, (grid_lon, grid_lat), method="linear"
    )
    B_interp = griddata(
        (lon_flat, lat_flat), B_flat, (grid_lon, grid_lat), method="linear"
    )

    rgb_image = np.stack((R_interp, G_interp, B_interp), axis=-1)
    rgb_max = np.nanmax(rgb_image)
    if not np.isfinite(rgb_max) or rgb_max == 0:
        rgb_max = 1.0
    rgb_image = np.clip((rgb_image / rgb_max) * exposure_coefficient, 0, 1)
    extent_raw = [lon_min, lon_max, lat_bot, lat_top]

    interp_output = griddata(
        (final_output[:, 1], final_output[:, 0]),
        final_output[:, 2],
        (grid_lon, grid_lat),
        method="linear",
    )
    interp_output = np.ma.masked_invalid(interp_output)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image, extent=extent_raw, origin="upper")
    im = plt.imshow(
        interp_output,
        extent=extent_raw,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label('(mg m$^{-3}$)', fontsize=16)
    plt.title(f"{structure_name}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {npy_path} (for npy_to_tif)")

    # Return numpy array for direct use
    return final_output


def preprocess_emit_data_Robust(
    nc_path: str,
    scaler_Rrs: Any,
    use_diff: bool = False,
    full_band_wavelengths: Optional[List[float]] = None,
) -> Tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess EMIT NetCDF data using robust scaling for model inference.

    This function reads EMIT L2 data from NetCDF, extracts specified wavelength
    bands, applies filtering and masking, optionally performs spectral smoothing
    and differencing, and returns a DataLoader ready for inference.

    Args:
        nc_path: Path to EMIT L2 NetCDF file.
        scaler_Rrs: Pre-fitted robust scaler from training.
        use_diff: Whether to apply Gaussian smoothing and first-order differencing.
        full_band_wavelengths: List of wavelengths (nm) to extract. If a band is
            not present, the closest available band is used.

    Returns:
        test_loader: DataLoader containing preprocessed data for inference.
        filtered_Rrs: Array of extracted Rrs bands with shape (H, W, B).
        mask: Boolean mask indicating valid pixels with shape (H, W).
        latitude: Array of latitude values.
        longitude: Array of longitude values.

    Raises:
        ValueError: If full_band_wavelengths is not provided or no Rrs bands are found.
    """

    if full_band_wavelengths is None:
        raise ValueError(
            "full_band_wavelengths must be provided to match EMIT Rrs bands"
        )

    def find_closest_band(target, available_bands):
        rrs_bands = [b for b in available_bands if b.startswith("Rrs_")]
        available_waves = [int(b.split("_")[1]) for b in rrs_bands]
        if not available_waves:
            raise ValueError("❌ No Rrs_* bands found in dataset")
        closest_wave = min(available_waves, key=lambda w: abs(w - target))
        return f"Rrs_{closest_wave}"

    dataset = Dataset(nc_path)
    latitude = dataset.variables["lat"][:]
    longitude = dataset.variables["lon"][:]

    all_vars = dataset.variables.keys()

    bands_to_extract = []
    for w in full_band_wavelengths:
        band_name = f"Rrs_{int(w)}"
        if band_name in all_vars:
            bands_to_extract.append(band_name)
        else:
            closest = find_closest_band(int(w), all_vars)
            print(f"⚠️ {band_name} does not exist, using the closest band {closest}")
            bands_to_extract.append(closest)
    filtered_Rrs = np.array([dataset.variables[band][:] for band in bands_to_extract])
    filtered_Rrs = np.moveaxis(filtered_Rrs, 0, -1)

    mask = np.all(~np.isnan(filtered_Rrs), axis=2)

    target_443 = (
        f"Rrs_443"
        if "Rrs_443" in bands_to_extract
        else find_closest_band(443, bands_to_extract)
    )
    target_560 = (
        f"Rrs_560"
        if "Rrs_560" in bands_to_extract
        else find_closest_band(560, bands_to_extract)
    )

    print(f"Using {target_443} and {target_560} for mask check.")

    idx_443 = bands_to_extract.index(target_443)
    idx_560 = bands_to_extract.index(target_560)
    mask &= filtered_Rrs[:, :, idx_443] <= filtered_Rrs[:, :, idx_560]

    valid_test_data = filtered_Rrs[mask]

    # ---- smooth + diff
    if use_diff:
        from scipy.ndimage import gaussian_filter1d

        Rrs_smoothed = np.array(
            [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
        )
        Rrs_processed = np.diff(Rrs_smoothed, axis=1)
        print("✅ [5] Performed Gaussian smoothing + first-order differencing")
    else:
        Rrs_processed = valid_test_data
        print("✅ [5] Smoothing and differencing not enabled")

    # ---- normalize
    Rrs_normalized = scaler_Rrs.transform(
        torch.tensor(Rrs_processed, dtype=torch.float32)
    ).numpy()

    # ---- DataLoader
    test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
    test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)
    print("✅ [6] DataLoader construction completed")

    return test_loader, filtered_Rrs, mask, latitude, longitude


def preprocess_emit_data_minmax(
    nc_path: str,
    full_band_wavelengths: Optional[List[float]] = None,
    diff_before_norm: bool = False,
    diff_after_norm: bool = False,
) -> Optional[Tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess EMIT NetCDF data using sample-wise MinMax scaling for inference.

    This function reads EMIT L2 data from NetCDF, extracts specified wavelength
    bands, applies filtering and masking, optionally performs spectral smoothing
    and differencing, applies sample-wise MinMax normalization, and returns a
    DataLoader ready for inference.

    Args:
        nc_path: Path to EMIT L2 NetCDF file.
        full_band_wavelengths: List of wavelengths (nm) to extract. If a band is
            not present, the closest available band is used.
        diff_before_norm: Whether to apply differencing before normalization.
        diff_after_norm: Whether to apply differencing after normalization.

    Returns:
        test_loader: DataLoader containing preprocessed data for inference.
        Rrs: Array of extracted Rrs bands with shape (H, W, B).
        mask: Boolean mask indicating valid pixels with shape (H, W).
        latitude: Array of latitude values.
        longitude: Array of longitude values.
        Returns None if an error occurs during processing.

    Raises:
        ValueError: If full_band_wavelengths is empty, no Rrs bands are found,
            or no valid pixels pass filtering.
    """
    print(f"📥 Start processing: {nc_path}")

    # ---- sanity checks
    if full_band_wavelengths is None or len(full_band_wavelengths) == 0:
        raise ValueError(
            "A non-empty full_band_wavelengths must be provided (e.g., [400, 402, ...])."
        )

    full_band_wavelengths = [int(w) for w in full_band_wavelengths]

    try:
        with Dataset(nc_path) as dataset:
            latitude = dataset.variables["lat"][:]
            longitude = dataset.variables["lon"][:]
            all_vars = set(dataset.variables.keys())
            available_wavelengths = [
                float(v.split("_")[1]) for v in all_vars if v.startswith("Rrs_")
            ]

            def find_closest_band(target_nm: float):
                nearest = min(available_wavelengths, key=lambda w: abs(w - target_nm))
                return f"Rrs_{int(nearest)}"

            # Search according to full_band_wavelengths
            bands_to_extract = []
            for w in full_band_wavelengths:
                band_name = f"Rrs_{w}"
                if band_name in all_vars:
                    bands_to_extract.append(band_name)
                else:
                    closest = find_closest_band(w)
                    print(
                        f"⚠️ {band_name} does not exist, using the closest band {closest}"
                    )
                    bands_to_extract.append(closest)

            seen = set()
            bands_to_extract = [
                b for b in bands_to_extract if not (b in seen or seen.add(b))
            ]

            if len(bands_to_extract) == 0:
                raise ValueError("❌ No usable Rrs_* bands found in the file.")
            # ---- read and stack to (H, W, B)
            # Each variable expected shape: (lat, lon) or (y, x)
            Rrs_stack = []
            for band in bands_to_extract:
                arr = dataset.variables[band][:]  # (H, W)
                Rrs_stack.append(arr)

            Rrs = np.array(Rrs_stack)  # (B, H, W)
            Rrs = np.moveaxis(Rrs, 0, -1)  # (H, W, B)
            filtered_Rrs = Rrs  # keep naming consistent with your previous return

            # ---- build mask using 440 & 560 (or nearest present within your requested list)
            have_waves = [int(b.split("_")[1]) for b in bands_to_extract]

            def nearest_idx(target_nm: int):
                # find nearest *among bands_to_extract*
                nearest_w = min(have_waves, key=lambda w: abs(w - target_nm))
                return bands_to_extract.index(f"Rrs_{nearest_w}")

            # Prefer exact if available; otherwise nearest in the user-requested set
            idx_440 = (
                bands_to_extract.index("Rrs_440")
                if "Rrs_440" in bands_to_extract
                else nearest_idx(440)
            )
            idx_560 = (
                bands_to_extract.index("Rrs_560")
                if "Rrs_560" in bands_to_extract
                else nearest_idx(560)
            )

            print(
                f"✅ Bands used for mask check: {bands_to_extract[idx_440]} and {bands_to_extract[idx_560]}"
            )

            mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)
            mask_condition = filtered_Rrs[:, :, idx_560] >= filtered_Rrs[:, :, idx_440]
            mask = mask_nanfree & mask_condition
            print(f"✅ [4] Built valid mask, remaining pixels: {int(np.sum(mask))}")

            if not np.any(mask):
                raise ValueError("❌ No valid pixels passed the filtering.")

            valid_test_data = filtered_Rrs[mask]  # (N, B)

        # === Check whether smoothing is needed (only executed if any differencing is enabled) ===
        if diff_before_norm or diff_after_norm:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_test_data]
            )
            print("✅ Gaussian smoothing applied")
        else:
            Rrs_smoothed = valid_test_data
            print("✅ Smoothing not enabled")

        # === Preprocessing before differencing ===
        if diff_before_norm:
            Rrs_preprocessed = np.diff(Rrs_smoothed, axis=1)
            print("✅ Preprocessing before differencing completed")
        else:
            Rrs_preprocessed = Rrs_smoothed
            print("✅ Preprocessing before differencing not enabled")

        # === MinMax normalization to [1, 10] ===
        scalers = [MinMaxScaler((1, 10)) for _ in range(Rrs_preprocessed.shape[0])]
        Rrs_normalized = np.array(
            [
                scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
                for i, row in enumerate(Rrs_preprocessed)
            ]
        )

        # === Post-processing after differencing ===
        if diff_after_norm:
            Rrs_normalized = np.diff(Rrs_normalized, axis=1)
            print("✅ Post-processing after differencing completed")
        else:
            print("✅ Post-processing after differencing not enabled")

        # === Construct DataLoader
        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

        return test_loader, Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ [ERROR] Failed to process file {nc_path}: {e}")
        return None


def npy_to_tif(
    npy_input: Union[str, np.ndarray],
    out_tif: str,
    resolution_m: float = 30,
    method: str = "linear",
    nodata_val: float = -9999.0,
    bbox_padding: float = 0.0,
    lat_col: int = 0,
    lon_col: int = 1,
    band_cols: Optional[Union[int, List[int]]] = None,
    band_names: Optional[List[str]] = None,
    wavelengths: Optional[List[float]] = None,
    crs: str = "EPSG:4326",
    compress: str = "deflate",
    bigtiff: str = "IF_SAFER",
) -> None:
    """Convert scattered point data to a multi-band GeoTIFF.

    This function takes point data in the format [lat, lon, band1, band2, ...]
    and interpolates it onto a regular grid to create a georeferenced GeoTIFF file.

    Args:
        npy_input: Path to .npy file or numpy array of shape [N, M] where N is
            the number of points and M includes lat, lon, and band values.
        out_tif: Output path for the GeoTIFF file.
        resolution_m: Spatial resolution in meters.
        method: Interpolation method ('linear', 'nearest', 'cubic').
        nodata_val: Value to use for NoData pixels.
        bbox_padding: Padding to add to bounding box in degrees.
        lat_col: Column index containing latitude values.
        lon_col: Column index containing longitude values.
        band_cols: Column indices to rasterize as bands. If None, uses all columns
            except lat/lon.
        band_names: Optional list of band description names.
        wavelengths: Optional list of wavelengths for band descriptions (e.g., [440, 619]).
        crs: Coordinate reference system string.
        compress: Compression method for GeoTIFF.
        bigtiff: BigTIFF creation option ('YES', 'NO', 'IF_NEEDED', 'IF_SAFER').

    Raises:
        TypeError: If npy_input is neither a path string nor numpy array.
        ValueError: If input array has fewer than 3 columns or no bands are selected.
    """

    # --- 1) Load data ---
    if isinstance(npy_input, str):
        arr = np.load(npy_input)
    elif isinstance(npy_input, np.ndarray):
        arr = npy_input
    else:
        raise TypeError("npy_input must be either a path string or a numpy.ndarray.")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Input must be 2D with >=3 columns (lat, lon, values...).")

    lat = arr[:, lat_col].astype(float)
    lon = arr[:, lon_col].astype(float)

    # --- 2) Band selection ---
    if band_cols is None:
        band_cols = [i for i in range(arr.shape[1]) if i not in (lat_col, lon_col)]
    if isinstance(band_cols, (int, np.integer)):
        band_cols = [int(band_cols)]
    if len(band_cols) == 0:
        raise ValueError("No value columns selected for bands.")

    # --- 3) Bounds (+ padding) ---
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min -= bbox_padding
    lat_max += bbox_padding
    lon_min -= bbox_padding
    lon_max += bbox_padding

    # --- 4) Resolution conversion ---
    lat_center = (lat_min + lat_max) / 2.0
    deg_per_m_lat = 1.0 / 111000.0
    deg_per_m_lon = 1.0 / (111000.0 * np.cos(np.radians(lat_center)))
    res_lat_deg = resolution_m * deg_per_m_lat
    res_lon_deg = resolution_m * deg_per_m_lon

    # --- 5) Grid ---
    lon_axis = np.arange(lon_min, lon_max + res_lon_deg, res_lon_deg)
    lat_axis = np.arange(lat_min, lat_max + res_lat_deg, res_lat_deg)
    Lon, Lat = np.meshgrid(lon_axis, lat_axis)

    transform = from_origin(lon_axis.min(), lat_axis.max(), res_lon_deg, res_lat_deg)

    # --- 6) Interpolation ---
    grids = []
    for idx in band_cols:
        vals = arr[:, idx].astype(float)

        g = griddata(points=(lon, lat), values=vals, xi=(Lon, Lat), method=method)
        if np.isnan(g).any():
            g_near = griddata(
                points=(lon, lat), values=vals, xi=(Lon, Lat), method=method
            )
            g = np.where(np.isnan(g), g_near, g)

        grids.append(np.flipud(g).astype(np.float32))

    data_stack = np.stack(grids, axis=0)

    # --- 7) Write GeoTIFF ---
    profile = {
        "driver": "GTiff",
        "height": data_stack.shape[1],
        "width": data_stack.shape[2],
        "count": data_stack.shape[0],
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_val,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": bigtiff,
    }

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        for b in range(data_stack.shape[0]):
            band = data_stack[b]
            band[~np.isfinite(band)] = nodata_val
            dst.write(band, b + 1)

        # Descriptions
        n_bands = data_stack.shape[0]
        if band_names is not None and len(band_names) == n_bands:
            descriptions = list(map(str, band_names))
        elif wavelengths is not None and len(wavelengths) == n_bands:
            descriptions = [f"aphy_{int(wl)}" for wl in wavelengths]
        else:
            descriptions = [f"band_{band_cols[b]}" for b in range(n_bands)]

        for b in range(1, n_bands + 1):
            dst.set_band_description(b, descriptions[b - 1])

    print(f"✅ GeoTIFF saved: {out_tif}")
