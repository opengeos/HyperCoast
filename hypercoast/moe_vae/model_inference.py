"""Model inference utilities for MoE-VAE.

This module provides functions for preprocessing and inference using MoE-VAE models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset as nc

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler
    from scipy.interpolate import griddata
    from torch.utils.data import DataLoader, TensorDataset
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    pass

from ..pace import read_pace


def preprocess_pace_data_Robust(
    nc_path, scaler_Rrs, use_diff=True, full_band_wavelengths=None
):
    """
    Preprocess PACE data for Robust scaling.

    Args:
        nc_path (str): Path to the NetCDF file containing PACE data.
        scaler_Rrs (object): RobustScaler object for Rrs normalization.
        use_diff (bool): Whether to apply first-order differencing.
        full_band_wavelengths (list): List of target wavelength bands.

    Returns:
        test_loader (DataLoader): DataLoader for test data.
        filtered_Rrs (array): Filtered Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
    """
    print(f"📥 Start processing: {nc_path}")
    try:
        PACE_dataset = read_pace(nc_path)
        print("✅ [1] Successfully read PACE data")

        da = PACE_dataset["Rrs"]
        Rrs = da.values  # [lat, lon, bands]
        latitude = da.latitude.values
        longitude = da.longitude.values
        print("✅ [2] Successfully retrieved Rrs, lat, and lon")

        # ✅ Extract wavelength
        if full_band_wavelengths is None:
            raise ValueError(
                "full_band_wavelengths must be provided to match PACE Rrs bands"
            )

        if hasattr(da, "wavelength") or "wavelength" in da.coords:
            pace_band_wavelengths = da.wavelength.values
        else:
            raise ValueError(
                "❌ Unable to extract wavelength from PACE data. Please check the NetCDF file structure."
            )

        missing = [b for b in full_band_wavelengths if b not in pace_band_wavelengths]
        if missing:
            raise ValueError(
                f"❌ The following wavelengths are not present in the PACE data: {missing}"
            )

        indices = [
            np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
        ]
        band_wavelengths = pace_band_wavelengths[indices]
        assert (
            band_wavelengths == np.array(full_band_wavelengths)
        ).all(), "❌ Band order mismatch"

        filtered_Rrs = Rrs[:, :, indices]
        print(
            f"✅ [3] Bands re-extracted based on full_band_wavelengths, total {len(indices)} bands"
        )

        # ✅ Build mask
        idx_440 = np.where(band_wavelengths == 440)[0][0]
        idx_560 = np.where(band_wavelengths == 560)[0][0]

        Rrs_440 = filtered_Rrs[:, :, idx_440]
        Rrs_560 = filtered_Rrs[:, :, idx_560]

        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)
        mask_condition = Rrs_560 >= Rrs_440
        mask = mask_nanfree & mask_condition
        print(f"✅ [4] Built valid mask, remaining pixels: {np.sum(mask)}")

        if not np.any(mask):
            raise ValueError("❌ No valid pixels passed the filtering.")

        valid_test_data = filtered_Rrs[mask]

        # ✅ Smoothing before differencing (enabled only if use_diff=True)
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

        # ✅ Normalization (RobustScaler provided)
        Rrs_normalized = scaler_Rrs.transform(
            torch.tensor(Rrs_processed, dtype=torch.float32)
        ).numpy()

        # ✅ Construct DataLoader
        from torch.utils.data import DataLoader, TensorDataset

        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)
        print("✅ [6] DataLoader construction completed")

        return test_loader, filtered_Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ [ERROR] Failed to process file {nc_path}: {e}")
        return None


def infer_and_visualize_single_model_Robust(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    TSS_scalers_dict,
    run=None,
    vmin=0,
    vmax=50,
):
    """
    Infer and visualize results from a single MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_loader (DataLoader): DataLoader for test data.
        Rrs (array): Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
        save_folder (str): Path to save the results.
        extent (tuple): Tuple containing the extent of the image.
        rgb_image (array): RGB image.
        structure_name (str): Name of the structure.
        TSS_scalers_dict (dict): Dictionary containing the scalers for TSS.
        run (int, optional): Run number. Defaults to None.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.

    Returns:
        numpy.ndarray: Final output array with lat, lon, and predicted values.
    """
    device = next(model.parameters()).device
    predictions_all = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            # === Inverse transform using TSS_scalers_dict from training ===
            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(predictions.cpu().numpy(), dtype=torch.float32)
            )
            predictions_all.append(
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )

    predictions_all = np.vstack(predictions_all).squeeze(-1)
    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all
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

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    # cbar.set_label("(mg m$^{-3}$)", fontsize=16)
    title = f"{structure_name}" if run is None else f"{structure_name} - Run {run}"
    plt.title(title, loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

    return final_output


def infer_and_visualize_token_model_Robust(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    run,
    TSS_scalers_dict,
    vmin=0,
    vmax=50,
):
    """
    Infer and visualize results from a token-based MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_loader (DataLoader): DataLoader for test data.
        Rrs (array): Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
        save_folder (str): Path to save the results.
        extent (tuple): Tuple containing the extent of the image.
        rgb_image (array): RGB image.
        structure_name (str): Name of the structure.
        run (int): Run number.
        TSS_scalers_dict (dict): Dictionary containing the scalers for TSS.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
    """
    device = next(model.parameters()).device
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]  # shape [B, token_len]

            # === Aggregate by token ===
            if predictions.ndim == 2:
                predictions = predictions.mean(dim=1, keepdim=True)  # [B, 1]

            # === Robust + log inverse transform ===
            predictions_log = TSS_scalers_dict["robust"].inverse_transform(
                torch.tensor(predictions.cpu().numpy(), dtype=torch.float32)
            )
            predictions_all.append(
                TSS_scalers_dict["log"].inverse_transform(predictions_log).numpy()
            )

    # === Concatenate and remove extra dimensions ===
    predictions_all = np.vstack(predictions_all).reshape(-1)

    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))

    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("(mg m$^{-3}$)", fontsize=16)
    plt.title(f"{structure_name} - Run {run}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def preprocess_pace_data_minmax(
    nc_path, full_band_wavelengths=None, diff_before_norm=False, diff_after_norm=False
):
    """
    Preprocess PACE data for MinMax scaling.

    Args:
        nc_path (str): Path to the NetCDF file containing PACE data.
        full_band_wavelengths (list): List of target wavelength bands.
        diff_before_norm (bool): Whether to apply first-order differencing before normalization.
        diff_after_norm (bool): Whether to apply first-order differencing after normalization.

    Returns:
        test_loader (DataLoader): DataLoader for test data.
        Rrs (array): Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
    """
    try:
        # === Load data ===
        PACE_dataset = read_pace(nc_path)
        da = PACE_dataset["Rrs"]
        Rrs = da.values  # [lat, lon, bands]
        latitude = da.latitude.values
        longitude = da.longitude.values

        # === Band check ===
        if full_band_wavelengths is None:
            raise ValueError(
                "full_band_wavelengths must be provided to match PACE Rrs bands"
            )

        if hasattr(da, "wavelength") or "wavelength" in da.coords:
            pace_band_wavelengths = da.wavelength.values
        else:
            raise ValueError(
                "❌ Unable to extract wavelength from PACE data. Please check the NetCDF file structure."
            )

        # Check for missing bands
        missing = [b for b in full_band_wavelengths if b not in pace_band_wavelengths]
        if missing:
            raise ValueError(
                f"❌ The following wavelengths are not found in the PACE data: {missing}"
            )

        # Extract indices in the order of full_band_wavelengths
        indices = [
            np.where(pace_band_wavelengths == b)[0][0] for b in full_band_wavelengths
        ]
        band_wavelengths = pace_band_wavelengths[indices]
        assert (
            band_wavelengths == np.array(full_band_wavelengths)
        ).all(), "❌ Band order is inconsistent"

        # Extract Rrs for selected_bands
        filtered_Rrs = Rrs[:, :, indices]

        # === Mask construction ===
        idx_440 = np.where(band_wavelengths == 440)[0][0]
        idx_560 = np.where(band_wavelengths == 560)[0][0]
        Rrs_440 = filtered_Rrs[:, :, idx_440]
        Rrs_560 = filtered_Rrs[:, :, idx_560]

        mask_nanfree = np.all(~np.isnan(filtered_Rrs), axis=2)
        mask_condition = Rrs_560 >= Rrs_440
        mask = mask_nanfree & mask_condition
        if not np.any(mask):
            raise ValueError("❌ No valid pixels passed the filtering.")

        valid_data = filtered_Rrs[mask]  # [num_pixel, num_band]

        # === Check if smoothing is needed (only executed when any differencing is enabled) ===
        if diff_before_norm or diff_after_norm:
            from scipy.ndimage import gaussian_filter1d

            Rrs_smoothed = np.array(
                [gaussian_filter1d(spectrum, sigma=1) for spectrum in valid_data]
            )
            print("✅ Gaussian smoothing applied")
        else:
            Rrs_smoothed = valid_data
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

        # === Construct DataLoader ===
        test_tensor = TensorDataset(torch.tensor(Rrs_normalized).float())
        test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

        return test_loader, Rrs, mask, latitude, longitude

    except Exception as e:
        print(f"❌ [ERROR] Failed to process file {nc_path}: {e}")
        return None


def infer_and_visualize_single_model_minmax(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    run=None,
    vmin=0,
    vmax=50,
    log_offset=0.01,
):
    """
    Infer and visualize results from a single MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_loader (DataLoader): DataLoader for test data.
        Rrs (array): Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
        save_folder (str): Path to save the results.
        extent (tuple): Tuple containing the extent of the image.
        rgb_image (array): RGB image.
        structure_name (str): Name of the structure.
        run (int, optional): Run number. Defaults to None.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
        log_offset (float): Log offset for predictions.

    Returns:
        numpy.ndarray: Final output array with lat, lon, and predicted values.
    """
    device = next(model.parameters()).device
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]

            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    predictions_all = np.vstack(predictions_all).squeeze(-1)

    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

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

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("(mg m$^{-3}$)", fontsize=16)
    title = f"{structure_name}" if run is None else f"{structure_name} - Run {run}"
    plt.title(title, loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()

    return final_output


def infer_and_visualize_token_model_minmax(
    model,
    test_loader,
    Rrs,
    mask,
    latitude,
    longitude,
    save_folder,
    extent,
    rgb_image,
    structure_name,
    run,
    vmin=0,
    vmax=50,
    log_offset=0.01,
):
    """
    Infer and visualize results from a token-based MoE-VAE model.

    Args:
        model (torch.nn.Module): MoE-VAE model.
        test_loader (DataLoader): DataLoader for test data.
        Rrs (array): Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
        save_folder (str): Path to save the results.
        extent (tuple): Tuple containing the extent of the image.
        rgb_image (array): RGB image.
        structure_name (str): Name of the structure.
        run (int): Run number.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
        log_offset (float): Log offset for predictions.
    """
    device = next(model.parameters()).device
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output_dict = model(batch)
            predictions = output_dict["pred_y"]  # shape [B, token_len]

            # === Aggregate along the token dimension ===
            if predictions.ndim == 2:
                predictions = predictions.mean(dim=1, keepdim=True)  # shape [B, 1]

            predictions_np = predictions.cpu().numpy()
            predictions_original = (10**predictions_np) - log_offset
            predictions_all.append(predictions_original)

    # === Concatenate and flatten ===
    predictions_all = np.vstack(predictions_all).reshape(-1)

    outputs = np.full((Rrs.shape[0], Rrs.shape[1]), np.nan)
    outputs[mask] = predictions_all

    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))

    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(structure_name))[0]
    npy_path = os.path.join(save_folder, f"{base_name}.npy")
    png_path = os.path.join(save_folder, f"{base_name}.png")
    np.save(npy_path, final_output)

    latitude_masked = final_output[:, 0]
    longitude_masked = final_output[:, 1]
    tss_values = final_output[:, 2]

    mean_lat = (extent[2] + extent[3]) / 2
    resolution_deg_lat = 1000 / 111000
    resolution_deg_lon = 1000 / (111000 * np.cos(np.radians(mean_lat)))
    grid_lon = np.arange(extent[0], extent[1], resolution_deg_lon)
    grid_lat = np.arange(extent[3], extent[2], -resolution_deg_lat)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    tss_resampled = griddata(
        (longitude_masked, latitude_masked),
        tss_values,
        (grid_lon, grid_lat),
        method="linear",
    )
    tss_resampled = np.ma.masked_invalid(tss_resampled)

    plt.figure(figsize=(24, 6))
    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")
    im = plt.imshow(
        tss_resampled,
        extent=extent,
        cmap="jet",
        alpha=1,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("(mg m$^{-3}$)", fontsize=16)
    plt.title(f"{structure_name} - Run {run}", loc="left", fontsize=20)
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def preprocess_emit_data_Robust(
    nc_path, scaler_Rrs, use_diff=True, full_band_wavelengths=None
):
    """
    Preprocess EMIT data for Robust scaling.

    Args:
        nc_path (str): Path to the NetCDF file containing EMIT data.
        scaler_Rrs (object): RobustScaler object for Rrs normalization.
        use_diff (bool): Whether to apply first-order differencing.
        full_band_wavelengths (list): List of target wavelength bands.

    Returns:
        test_loader (DataLoader): DataLoader for test data.
        filtered_Rrs (array): Filtered Rrs data.
        mask (array): Boolean mask indicating valid pixels.
        latitude (array): Latitude values.
        longitude (array): Longitude values.
    """

    if full_band_wavelengths is None:
        raise ValueError(
            "full_band_wavelengths must be provided to match EMIT Rrs bands"
        )

    def find_closest_band(target, available_bands):
        available_waves = [int(b.split("_")[1]) for b in available_bands]
        closest_wave = min(available_waves, key=lambda w: abs(w - target))
        return f"Rrs_{closest_wave}"

    dataset = nc(nc_path)
    latitude = dataset.variables["lat"][:]
    longitude = dataset.variables["lon"][:]

    all_vars = dataset.variables.keys()

    bands_to_extract = []
    for w in full_band_wavelengths:
        band_name = f"Rrs_{int(w)}"
        if band_name in all_vars:
            bands_to_extract.append(band_name)
        else:
            closest = find_closest_band(int(w))
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
    nc_path, full_band_wavelengths=None, diff_before_norm=False, diff_after_norm=False
):
    """
    Read EMIT NetCDF, extract Rrs_* bands according to full_band_wavelengths,
    apply (optional) smooth+diff and robust normalization, and return a DataLoader.

    Args:
        nc_path (str): Path to the NetCDF file containing EMIT data.
        full_band_wavelengths (list): List of target wavelength bands.
        diff_before_norm (bool): Whether to apply first-order differencing before normalization.
        diff_after_norm (bool): Whether to apply first-order differencing after normalization.

    Returns:
        test_loader, filtered_Rrs(H, W, B), mask(H, W), latitude(H), longitude(W)
    """
    print(f"📥 Start processing: {nc_path}")

    # ---- sanity checks
    if full_band_wavelengths is None or len(full_band_wavelengths) == 0:
        raise ValueError(
            "A non-empty full_band_wavelengths must be provided (e.g., [400, 402, ...])."
        )

    full_band_wavelengths = [int(w) for w in full_band_wavelengths]

    try:
        with nc(nc_path, "r") as dataset:
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
    npy_input,
    out_tif,
    resolution_m=1000,
    method="linear",
    nodata_val=-9999.0,
    bbox_padding=0.0,
    lat_col=0,
    lon_col=1,
    band_cols=None,
    band_names=None,
    wavelengths=None,
    crs="EPSG:4326",
    compress="deflate",
    bigtiff="IF_SAFER",
):
    """Convert [lat, lon, band1, band2, ...] scattered points into a multi-band GeoTIFF.

    This function takes scattered point data with latitude, longitude, and one or more
    value columns, and interpolates them onto a regular grid to create a GeoTIFF file.

    Args:
        npy_input (str or np.ndarray): Path to .npy file or array of shape [N, M].
        out_tif (str): Output path for the GeoTIFF file.
        resolution_m (float, optional): Grid resolution in meters. Defaults to 1000.
        method (str, optional): Interpolation method ('linear', 'nearest', 'cubic').
            Defaults to 'linear'.
        nodata_val (float, optional): Value to use for missing data. Defaults to -9999.0.
        bbox_padding (float, optional): Padding to add to bounding box in degrees.
            Defaults to 0.0.
        lat_col (int, optional): Column index for latitude. Defaults to 0.
        lon_col (int, optional): Column index for longitude. Defaults to 1.
        band_cols (list, optional): Column indices to rasterize as bands.
            Defaults to all columns after lat/lon.
        band_names (list, optional): Band descriptions. Defaults to None.
        wavelengths (list, optional): Wavelengths for band descriptions.
            Defaults to None.
        crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".
        compress (str, optional): Compression method. Defaults to "deflate".
        bigtiff (str, optional): BigTIFF setting. Defaults to "IF_SAFER".

    Raises:
        TypeError: If npy_input is neither a path string nor a numpy array.
        ValueError: If input array has wrong shape or no value columns.
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
            descriptions = [f"band_{int(wl)}" for wl in wavelengths]
        else:
            descriptions = [f"band_{band_cols[b]}" for b in range(n_bands)]

        for b in range(1, n_bands + 1):
            dst.set_band_description(b, descriptions[b - 1])

    print(f"✅ GeoTIFF saved: {out_tif}")
