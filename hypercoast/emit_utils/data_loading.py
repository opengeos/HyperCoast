"""Data loading utilities for EMIT model training and inference.

This module provides functions for loading and preprocessing hyperspectral remote
sensing data from Excel files for training and testing machine learning models.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from .preprocess import LogScaler, RobustMinMaxScaler
except ImportError:
    pass


def load_real_data_Robust(
    excel_path: str,
    selected_bands: List[float],
    target_parameter: str = "TSS",
    split_ratio: float = 0.7,
    seed: int = 42,
    use_diff: bool = False,
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
    Rrs_range: Tuple[float, float] = (0, 0.25),
    target_range: Tuple[float, float] = (-0.5, 0.5),
) -> Tuple[DataLoader, DataLoader, int, int, List[str], List[str], Any, Dict[str, Any]]:
    """Load and preprocess real data using robust scaling for training and testing.

    This function reads hyperspectral remote sensing reflectance (Rrs) data and
    water quality parameters from an Excel file, applies robust scaling and
    preprocessing, and returns DataLoaders ready for model training.

    Args:
        excel_path: Path to Excel file containing 'Rrs' and 'parameter' sheets.
        selected_bands: List of wavelengths (nm) to extract from the data.
        target_parameter: Name of target parameter column (e.g., 'TSS', 'Chla').
        split_ratio: Proportion of data to use for training (0-1).
        seed: Random seed for reproducible train/test split.
        use_diff: Whether to apply first-order differencing to Rrs spectra.
        lower_quantile: Lower quantile for filtering target parameter outliers.
        upper_quantile: Upper quantile for filtering target parameter outliers.
        Rrs_range: Target range for Rrs normalization.
        target_range: Target range for parameter normalization.

    Returns:
        train_dl: DataLoader for training data.
        test_dl: DataLoader for testing data.
        input_dim: Number of input features.
        output_dim: Number of output features.
        train_ids: List of sample IDs in training set.
        test_ids: List of sample IDs in testing set.
        scaler_Rrs: Fitted scaler for Rrs data.
        TSS_scalers_dict: Dictionary containing 'log' and 'robust' scalers for target.
    """

    rounded_bands = [int(round(b)) for b in selected_bands]
    band_cols = [f"Rrs_{b}" for b in rounded_bands]

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    df_rrs_selected = df_rrs[["GLORIA_ID"] + band_cols]
    df_param_selected = df_param[["GLORIA_ID", target_parameter]]
    df_merged = pd.merge(
        df_rrs_selected, df_param_selected, on="GLORIA_ID", how="inner"
    )

    mask_rrs_valid = df_merged[band_cols].notna().all(axis=1)
    mask_param_valid = df_merged[target_parameter].notna()
    df_filtered = df_merged[mask_rrs_valid & mask_param_valid].reset_index(drop=True)

    print(
        f"Number of samples after filtering Rrs and {target_parameter}: {len(df_filtered)}"
    )

    lower = df_filtered[target_parameter].quantile(lower_quantile)
    top = df_filtered[target_parameter].quantile(upper_quantile)
    df_filtered = df_filtered[
        (df_filtered[target_parameter] >= lower)
        & (df_filtered[target_parameter] <= top)
    ].reset_index(drop=True)

    print(
        f"Number of samples after removing {target_parameter} quantiles [{lower_quantile}, {upper_quantile}]: {len(df_filtered)}"
    )

    all_sample_ids = df_filtered["GLORIA_ID"].astype(str).tolist()
    Rrs_array = df_filtered[band_cols].values
    param_array = df_filtered[[target_parameter]].values

    if use_diff:
        Rrs_array = np.diff(Rrs_array, axis=1)

    scaler_Rrs = RobustMinMaxScaler(feature_range=Rrs_range)
    scaler_Rrs.fit(torch.tensor(Rrs_array, dtype=torch.float32))
    Rrs_normalized = scaler_Rrs.transform(
        torch.tensor(Rrs_array, dtype=torch.float32)
    ).numpy()

    log_scaler = LogScaler(shift_min=False, safety_term=1e-8)
    param_log = log_scaler.fit_transform(torch.tensor(param_array, dtype=torch.float32))
    param_scaler = RobustMinMaxScaler(
        feature_range=target_range, global_scale=True, robust=True
    )
    param_transformed = param_scaler.fit_transform(param_log).numpy()

    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    param_tensor = torch.tensor(param_transformed, dtype=torch.float32)
    dataset = TensorDataset(Rrs_tensor, param_tensor)

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(split_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_ids = [all_sample_ids[i] for i in train_indices]
    test_ids = [all_sample_ids[i] for i in test_indices]

    train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = param_tensor.shape[1]
    TSS_scalers_dict = {"log": log_scaler, "robust": param_scaler}

    return (
        train_dl,
        test_dl,
        input_dim,
        output_dim,
        train_ids,
        test_ids,
        scaler_Rrs,
        TSS_scalers_dict,
    )


def load_real_test_Robust(
    excel_path: str,
    selected_bands: List[float],
    max_allowed_diff: float = 1.0,
    scaler_Rrs: Optional[Any] = None,
    scalers_dict: Optional[Dict[str, Any]] = None,
    use_diff: bool = False,
    target_parameter: str = "SPM",
) -> Tuple[DataLoader, int, int, List[str], List[str]]:
    """Load and preprocess test data using pre-fitted robust scalers.

    This function loads test data from Excel and applies the same preprocessing
    transformations used during training, including band matching, filtering,
    and scaling using pre-fitted scalers.

    Args:
        excel_path: Path to Excel file containing 'Rrs' and 'parameter' sheets.
        selected_bands: List of wavelengths (nm) to extract from the data.
        max_allowed_diff: Maximum wavelength difference (nm) allowed for band matching.
        scaler_Rrs: Pre-fitted scaler for Rrs data from training.
        scalers_dict: Dictionary containing 'log' and 'robust' scalers from training.
        use_diff: Whether to apply first-order differencing to Rrs spectra.
        target_parameter: Name of target parameter column (e.g., 'SPM', 'TSS').

    Returns:
        test_dl: DataLoader for test data.
        input_dim: Number of input features.
        output_dim: Number of output features (always 1).
        sample_ids: List of sample IDs in test set.
        sample_dates: List of sample dates in test set.

    Raises:
        ValueError: If row counts don't match, no valid samples exist, or band
            matching fails.
    """

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if df_rrs.shape[0] != df_param.shape[0]:
        raise ValueError(
            f"❌ The number of rows in the Rrs table and parameter table do not match. Rrs: {df_rrs.shape[0]}, parameter: {df_param.shape[0]}"
        )

    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()

    # Match target bands
    rrs_wavelengths = []
    rrs_cols = []
    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except:
            continue

    band_cols = []
    for target_band in selected_bands:
        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)
        if min_diff > max_allowed_diff:
            raise ValueError(
                f"Target wavelength {target_band} nm cannot be matched, error {min_diff:.2f} nm exceeds the allowed range"
            )
        best_idx = diffs.index(min_diff)
        band_cols.append(rrs_cols[best_idx])

    print(f"\n✅ Band matching successful, {len(selected_bands)} target bands in total")
    print(f"Final number of valid test samples: {df_rrs.shape[0]}\n")

    Rrs_array = df_rrs[band_cols].values
    param_array = df_param[[target_parameter]].values.flatten()
    # === Key: Remove rows with NaN/Inf before differencing ===
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(param_array)
    mask_ok = mask_inputs_ok & mask_target_ok
    if not np.any(mask_ok):
        raise ValueError("❌ Valid samples = 0 (NaN/Inf found in input or target).")
    dropped = int(len(mask_ok) - mask_ok.sum())
    if dropped > 0:
        print(
            f"⚠️ Dropped {dropped} invalid samples (containing NaN/Inf) before differencing"
        )

    Rrs_array = Rrs_array[mask_ok]
    param_array = param_array[mask_ok]
    sample_ids = [sid for sid, keep in zip(sample_ids, mask_ok) if keep]
    sample_dates = [d for d, keep in zip(sample_dates, mask_ok) if keep]

    if use_diff:
        Rrs_array = np.diff(Rrs_array, axis=1)

    Rrs_tensor = torch.tensor(Rrs_array, dtype=torch.float32)
    Rrs_normalized = scaler_Rrs.transform(Rrs_tensor).numpy()

    log_scaler = scalers_dict["log"]
    robust_scaler = scalers_dict["robust"]
    param_log = log_scaler.transform(
        torch.tensor(param_array.reshape(-1, 1), dtype=torch.float32)
    )
    param_transformed = robust_scaler.transform(param_log).numpy()

    dataset = TensorDataset(
        torch.tensor(Rrs_normalized, dtype=torch.float32),
        torch.tensor(param_transformed.reshape(-1, 1), dtype=torch.float32),
    )
    test_dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = 1

    return test_dl, input_dim, output_dim, sample_ids, sample_dates


def load_real_data(
    excel_path: str,
    selected_bands: List[float],
    split_ratio: float = 0.7,
    seed: int = 42,
    diff_before_norm: bool = False,
    diff_after_norm: bool = False,
    target_parameter: str = "TSS",
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
    log_offset: float = 0.01,
) -> Tuple[DataLoader, DataLoader, int, int, List[str], List[str]]:
    """Load and preprocess real data using MinMax scaling for training and testing.

    This function reads hyperspectral Rrs data and water quality parameters from
    an Excel file, applies sample-wise MinMax scaling and log transformation,
    and returns DataLoaders ready for model training.

    Args:
        excel_path: Path to Excel file containing 'Rrs' and 'parameter' sheets.
        selected_bands: List of wavelengths (nm) to extract from the data.
        split_ratio: Proportion of data to use for training (0-1).
        seed: Random seed for reproducible train/test split.
        diff_before_norm: Whether to apply differencing before normalization.
        diff_after_norm: Whether to apply differencing after normalization.
        target_parameter: Name of target parameter column (e.g., 'TSS', 'Chla').
        lower_quantile: Lower quantile for filtering target parameter outliers.
        upper_quantile: Upper quantile for filtering target parameter outliers.
        log_offset: Offset added before log transformation to avoid log(0).

    Returns:
        train_dl: DataLoader for training data.
        test_dl: DataLoader for testing data.
        input_dim: Number of input features.
        output_dim: Number of output features.
        train_ids: List of sample IDs in training set.
        test_ids: List of sample IDs in testing set.
    """

    rounded_bands = [int(round(b)) for b in selected_bands]
    band_cols = [f"Rrs_{b}" for b in rounded_bands]
    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")
    df_rrs_selected = df_rrs[["GLORIA_ID"] + band_cols]
    df_param_selected = df_param[["GLORIA_ID", target_parameter]]
    df_merged = pd.merge(
        df_rrs_selected, df_param_selected, on="GLORIA_ID", how="inner"
    )

    # === Filter valid samples ===
    mask_rrs_valid = df_merged[band_cols].notna().all(axis=1)
    mask_target_valid = df_merged[target_parameter].notna()
    df_filtered = df_merged[mask_rrs_valid & mask_target_valid].reset_index(drop=True)
    print(
        f"✅ Number of samples after filtering Rrs and {target_parameter}: {len(df_filtered)}"
    )

    # === Quantile clipping for target parameter ===
    lower = df_filtered[target_parameter].quantile(lower_quantile)
    upper = df_filtered[target_parameter].quantile(upper_quantile)
    df_filtered = df_filtered[
        (df_filtered[target_parameter] >= lower)
        & (df_filtered[target_parameter] <= upper)
    ].reset_index(drop=True)
    print(
        f"✅ Number of samples after removing {target_parameter} quantiles [{lower_quantile}, {upper_quantile}]: {len(df_filtered)}"
    )

    # === Extract sample IDs, Rrs, and target parameter ===
    all_sample_ids = df_filtered["GLORIA_ID"].astype(str).tolist()
    Rrs_array = df_filtered[band_cols].values
    param_array = df_filtered[[target_parameter]].values

    if diff_before_norm:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # === Apply MinMax scaling to [1, 10] for each sample independently ===
    scalers_Rrs_real = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    if diff_after_norm:
        Rrs_normalized = np.diff(Rrs_normalized, axis=1)

    # === Transform target parameter to log10(param + log_offset) ===
    param_transformed = np.log10(param_array + log_offset)

    # === Build Dataset ===
    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    param_tensor = torch.tensor(param_transformed, dtype=torch.float32)
    dataset = TensorDataset(Rrs_tensor, param_tensor)

    # === Split into training and testing sets ===
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_size = int(split_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_ids = [all_sample_ids[i] for i in train_indices]
    test_ids = [all_sample_ids[i] for i in test_indices]

    train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = param_tensor.shape[1]

    return (train_dl, test_dl, input_dim, output_dim, train_ids, test_ids)


def load_real_test(
    excel_path: str,
    selected_bands: List[float],
    max_allowed_diff: float = 1.0,
    diff_before_norm: bool = False,
    diff_after_norm: bool = False,
    target_parameter: str = "TSS",
    log_offset: float = 0.01,
) -> Tuple[DataLoader, int, int, List[str], List[str]]:
    """Load and preprocess test data using sample-wise MinMax scaling.

    This function loads test data from Excel, performs band matching to the
    nearest available wavelengths, applies sample-wise MinMax scaling and log
    transformation, and returns a DataLoader for inference.

    Args:
        excel_path: Path to Excel file containing 'Rrs' and 'parameter' sheets.
        selected_bands: List of target wavelengths (nm) to extract.
        max_allowed_diff: Maximum wavelength difference (nm) allowed for band matching.
        diff_before_norm: Whether to apply differencing before normalization.
        diff_after_norm: Whether to apply differencing after normalization.
        target_parameter: Name of target parameter column (e.g., 'TSS', 'SPM').
        log_offset: Offset added before log transformation to avoid log(0).

    Returns:
        test_dl: DataLoader for test data.
        input_dim: Number of input features.
        output_dim: Number of output features (always 1).
        sample_ids: List of sample IDs in test set.
        sample_dates: List of sample dates in test set.

    Raises:
        ValueError: If row counts don't match, no valid samples exist, or band
            matching fails.
    """

    df_rrs = pd.read_excel(excel_path, sheet_name="Rrs")
    df_param = pd.read_excel(excel_path, sheet_name="parameter")

    if df_rrs.shape[0] != df_param.shape[0]:
        raise ValueError(
            f"❌ The number of rows in the Rrs table and parameter table do not match. Rrs: {df_rrs.shape[0]}, parameter: {df_param.shape[0]}"
        )

    # === Extract IDs and dates ===
    sample_ids = df_rrs["Site Label"].astype(str).tolist()
    sample_dates = df_rrs["Date"].astype(str).tolist()

    # === Match target bands ===
    rrs_wavelengths = []
    rrs_cols = []
    for col in df_rrs.columns:
        try:
            wl = float(col)
            rrs_wavelengths.append(wl)
            rrs_cols.append(col)
        except Exception:
            continue

    band_cols = []
    matched_bands = []
    for target_band in selected_bands:
        diffs = [abs(wl - target_band) for wl in rrs_wavelengths]
        min_diff = min(diffs)
        if min_diff > max_allowed_diff:
            raise ValueError(
                f"Target wavelength {target_band} nm cannot be matched, error {min_diff:.2f} nm exceeds the allowed range"
            )
        best_idx = diffs.index(min_diff)
        band_cols.append(rrs_cols[best_idx])
        matched_bands.append(rrs_wavelengths[best_idx])

    print(
        f"\n✅ Band matching successful, {len(selected_bands)} target bands in total, {len(band_cols)} columns actually extracted"
    )
    print(f"Original number of test samples: {df_rrs.shape[0]}\n")

    # === Extract Rrs and target parameter (without differencing for now) ===
    Rrs_array = df_rrs[band_cols].values.astype(float)
    target_array = df_param[[target_parameter]].values.astype(float).flatten()

    # === Key: Remove rows with NaN/Inf before differencing ===
    mask_inputs_ok = np.all(np.isfinite(Rrs_array), axis=1)
    mask_target_ok = np.isfinite(target_array)
    mask_ok = mask_inputs_ok & mask_target_ok
    if not np.any(mask_ok):
        raise ValueError("❌ No valid samples (NaN/Inf found in input or target).")
    dropped = int(len(mask_ok) - mask_ok.sum())
    if dropped > 0:
        print(
            f"⚠️ Dropped {dropped} invalid samples (containing NaN/Inf) before differencing"
        )

    Rrs_array = Rrs_array[mask_ok]
    target_array = target_array[mask_ok]
    sample_ids = [sid for sid, keep in zip(sample_ids, mask_ok) if keep]
    sample_dates = [d for d, keep in zip(sample_dates, mask_ok) if keep]

    # === Preprocessing before differencing (optional) ===
    if diff_before_norm:
        Rrs_array = np.diff(Rrs_array, axis=1)

    # === Apply MinMaxScaler to [1, 10] for each sample ===
    scalers_Rrs_test = [MinMaxScaler((1, 10)) for _ in range(Rrs_array.shape[0])]
    Rrs_normalized = np.array(
        [
            scalers_Rrs_test[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_array)
        ]
    )

    # === Post-processing after differencing (optional) ===
    if diff_after_norm:
        Rrs_normalized = np.diff(Rrs_normalized, axis=1)

    # === Transform target value to log10(x + log_offset) ===
    target_transformed = np.log10(target_array + log_offset)

    # === Construct DataLoader ===
    Rrs_tensor = torch.tensor(Rrs_normalized, dtype=torch.float32)
    target_tensor = torch.tensor(target_transformed.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(Rrs_tensor, target_tensor)
    test_dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

    input_dim = Rrs_tensor.shape[1]
    output_dim = target_tensor.shape[1]

    return test_dl, input_dim, output_dim, sample_ids, sample_dates
