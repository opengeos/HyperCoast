"""
Module for training and evaluating the VAE model for Chl-a concentration estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
import pandas as pd
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.interpolate import griddata


class VAE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initializes the VAE model with encoder and decoder layers.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()

        # encoder
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(64, 32)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),
            nn.Softplus(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input tensor into mean and log variance.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors.
        """
        x = self.encoder_layer(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): Mean tensor.
            log_var (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent vector back to the original space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the VAE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed tensor,
                mean, and log variance.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var


def loss_function(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Computes the VAE loss function.

    Args:
        recon_x (torch.Tensor): Reconstructed tensor.
        x (torch.Tensor): Original input tensor.
        mu (torch.Tensor): Mean tensor.
        log_var (torch.Tensor): Log variance tensor.

    Returns:
        torch.Tensor: Computed loss.
    """
    L1 = F.l1_loss(recon_x, x, reduction="mean")
    BCE = F.mse_loss(recon_x, x, reduction="mean")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return L1


def train(
    model: VAE,
    train_dl: DataLoader,
    epochs: int = 200,
    device: torch.device = None,
    opt: torch.optim.Optimizer = None,
    model_path: str = "model/vae_model_PACE.pth",
    best_model_path: str = "model/vae_trans_model_best_Chl_PACE.pth",
) -> None:
    """Trains the VAE model.

    Args:
        model (VAE): VAE model to be trained.
        train_dl (DataLoader): DataLoader for training data.
        epochs (int, optional): Number of epochs to train. Defaults to 200.
        device (torch.device, optional): Device to train on. Defaults to None.
        opt (torch.optim.Optimizer, optional): Optimizer. Defaults to None.
        model_path (str, optional): Path to save the model. Defaults to
            "model/vae_model_PACE.pth".
        best_model_path (str, optional): Path to save the best model. Defaults
            to "model/vae_trans_model_best_Chl_PACE.pth"
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    model.train()

    min_total_loss = float("inf")
    # Save the optimal model

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            y_pred, mu, log_var = model(x)
            loss = loss_function(y_pred, y, mu, log_var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f"epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}")

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_path)
    # Save the model from the last epoch.
    torch.save(model.state_dict(), model_path)


def evaluate(
    model: VAE, test_dl: DataLoader, device: torch.device = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the VAE model.

    Args:
        model (VAE): VAE model to be evaluated.
        test_dl (DataLoader): DataLoader for test data.
        device (torch.device, optional): Device to evaluate on. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predictions and actual values.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred, _, _ = model(x)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)


def load_real_data(
    aphy_file_path: str, rrs_file_path: str
) -> tuple[DataLoader, DataLoader, int, int]:
    """Loads and preprocesses real data for training and testing.

    Args:
        aphy_file_path (str): Path to the aphy file.
        rrs_file_path (str): Path to the rrs file.

    Returns:
        tuple[DataLoader, DataLoader, int, int]: Training DataLoader, testing
            DataLoader, input dimension, output dimension.
    """
    array1 = np.loadtxt(aphy_file_path, delimiter=",", dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=",", dtype=float)

    array1 = array1.reshape(-1, 1)

    Rrs_real = array2
    Chl_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [
        MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])
    ]

    Rrs_real_normalized = np.array(
        [
            scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_real)
        ]
    )

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    Chl_real_tensor = torch.tensor(Chl_real, dtype=torch.float32)

    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)

    train_size = int(0.7 * len(dataset_real))
    test_size = len(dataset_real) - train_size
    train_dataset_real, test_dataset_real = random_split(
        dataset_real, [train_size, test_size]
    )

    train_real_dl = DataLoader(
        train_dataset_real, batch_size=1024, shuffle=True, num_workers=12
    )
    test_real_dl = DataLoader(
        test_dataset_real, batch_size=1024, shuffle=False, num_workers=12
    )

    return train_real_dl, test_real_dl, input_dim, output_dim


def load_real_test(
    aphy_file_path: str, rrs_file_path: str
) -> tuple[DataLoader, int, int]:
    """Loads and preprocesses real data for testing.

    Args:
        aphy_file_path (str): Path to the aphy file.
        rrs_file_path (str): Path to the rrs file.

    Returns:
        tuple[DataLoader, int, int]: Testing DataLoader, input dimension, output dimension.
    """
    array1 = np.loadtxt(aphy_file_path, delimiter=",", dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=",", dtype=float)

    array1 = array1.reshape(-1, 1)

    Rrs_real = array2
    Chl_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = Chl_real.shape[1]

    scalers_Rrs_real = [
        MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])
    ]

    Rrs_real_normalized = np.array(
        [
            scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten()
            for i, row in enumerate(Rrs_real)
        ]
    )

    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    Chl_real_tensor = torch.tensor(Chl_real, dtype=torch.float32)

    dataset_real = TensorDataset(Rrs_real_tensor, Chl_real_tensor)
    dataset_size = int(len(dataset_real))
    test_real_dl = DataLoader(
        dataset_real, batch_size=dataset_size, shuffle=False, num_workers=12
    )

    return test_real_dl, input_dim, output_dim


def calculate_metrics(
    predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.8
) -> tuple[float, float, float, float, float, float, float]:
    """Calculates epsilon, beta, and additional metrics (RMSE, RMSLE, MAPE, Bias, MAE).

    Args:
        predictions (np.ndarray): Predicted values.
        actuals (np.ndarray): Actual values.
        threshold (float, optional): Relative error threshold. Defaults to 0.8.

    Returns:
        tuple[float, float, float, float, float, float, float]: epsilon, beta, rmse, rmsle, mape, bias, mae.
    """
    # Apply the threshold to filter out predictions with large relative error
    mask = np.abs(predictions - actuals) / np.abs(actuals + 1e-10) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / filtered_actuals)
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 50 * (10**Y - 1)
    beta = 50 * np.sign(Z) * (10 ** np.abs(Z) - 1)

    # Calculate additional metrics
    rmse = np.sqrt(np.mean((filtered_predictions - filtered_actuals) ** 2))
    rmsle = np.sqrt(
        np.mean(
            (np.log10(filtered_predictions + 1) - np.log10(filtered_actuals + 1)) ** 2
        )
    )
    mape = 50 * np.median(
        np.abs((filtered_predictions - filtered_actuals) / filtered_actuals)
    )
    bias = 10 ** (np.mean(np.log10(filtered_predictions) - np.log10(filtered_actuals)))
    mae = 10 ** np.mean(
        np.abs(np.log10(filtered_predictions) - np.log10(filtered_actuals))
    )

    return epsilon, beta, rmse, rmsle, mape, bias, mae


def plot_results(
    predictions_rescaled: np.ndarray,
    actuals_rescaled: np.ndarray,
    save_dir: str,
    threshold: float = 0.5,
    mode: str = "test",
) -> None:
    """Plots the results of the predictions against the actual values.

    Args:
        predictions_rescaled (np.ndarray): Rescaled predicted values.
        actuals_rescaled (np.ndarray): Rescaled actual values.
        save_dir (str): Directory to save the plot.
        threshold (float, optional): Relative error threshold. Defaults to 0.5.
        mode (str, optional): Mode of the plot (e.g., "test"). Defaults to "test".
    """

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    mask = np.abs(predictions - actuals) / np.abs(actuals + 1e-10) < 1
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    log_actual = np.log10(np.where(actuals == 0, 1e-10, actuals))
    log_prediction = np.log10(np.where(predictions == 0, 1e-10, predictions))

    filtered_log_actual = np.log10(
        np.where(filtered_actuals == 0, 1e-10, filtered_actuals)
    )
    filtered_log_prediction = np.log10(
        np.where(filtered_predictions == 0, 1e-10, filtered_predictions)
    )

    epsilon, beta, rmse, rmsle, mape, bias, mae = calculate_metrics(
        filtered_predictions, filtered_actuals, threshold
    )

    valid_mask = np.isfinite(filtered_log_actual) & np.isfinite(filtered_log_prediction)
    slope, intercept = np.polyfit(
        filtered_log_actual[valid_mask], filtered_log_prediction[valid_mask], 1
    )
    x = np.array([-2, 4])
    y = slope * x + intercept

    _ = plt.figure(figsize=(6, 6))

    plt.plot(x, y, linestyle="--", color="blue", linewidth=0.8)
    lims = [-2, 4]
    plt.plot(lims, lims, linestyle="-", color="black", linewidth=0.8)

    sns.scatterplot(x=log_actual, y=log_prediction, alpha=0.5)

    sns.kdeplot(
        x=filtered_log_actual,
        y=filtered_log_prediction,
        levels=3,
        color="black",
        fill=False,
        linewidths=0.8,
    )

    plt.xlabel("Actual $Chla$ Values", fontsize=16)
    plt.ylabel("Predicted $Chla$ Values", fontsize=16)
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.grid(True, which="both", ls="--")

    plt.legend(
        title=(
            f"MAE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n"
            f"Bias = {bias:.2f}, Slope = {slope:.2f} \n"
            f"MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=16,
        title_fontsize=12,
    )

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    plt.savefig(os.path.join(save_dir, f"{mode}_plot.pdf"), bbox_inches="tight")
    # plt.close()


def save_to_csv(data: np.ndarray, file_path: str) -> None:
    """Saves data to a CSV file.

    Args:
        data (np.ndarray): Data to be saved.
        file_path (str): Path to the CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def chla_predict(
    pace_filepath: str,
    best_model_path: str,
    chla_data_file: str = None,
    device: torch.device = None,
) -> None:
    """Predicts chlorophyll-a concentration using a pre-trained VAE model.

    Args:
        pace_filepath (str): Path to the PACE dataset file.
        best_model_path (str): Path to the pre-trained VAE model file.
        chla_data_file (str, optional): Path to save the predicted chlorophyll-a data. Defaults to None.
        device (torch.device, optional): Device to perform inference on. Defaults to None.
    """

    from .pace import read_pace

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PACE dataset and prepare data
    PACE_dataset = read_pace(pace_filepath)
    da = PACE_dataset["Rrs"]
    wl = da.wavelength.values
    Rrs = da.values
    latitude = da.latitude.values
    longitude = da.longitude.values

    # Filter wavelengths between 400 and 703 nm
    indices = np.where((wl >= 400) & (wl <= 703))[0]
    filtered_Rrs = Rrs[:, :, indices]

    # Save filtered Rrs and wavelength
    filtered_wl = wl[indices]

    # Create a mask that is 1 where all wavelengths for a given pixel have non-NaN values, and 0 otherwise
    mask = np.all(~np.isnan(filtered_Rrs), axis=2).astype(int)

    # Define input and output dimensions
    input_dim = 148
    output_dim = 1

    # Load test data and mask
    test_data = filtered_Rrs
    mask_data = mask

    # Filter valid data using the mask
    mask = mask_data == 1
    N = np.sum(mask)
    valid_test_data = test_data[mask]

    # Normalize data
    valid_test_data = np.array(
        [
            (
                MinMaxScaler(feature_range=(1, 10))
                .fit_transform(row.reshape(-1, 1))
                .flatten()
                if not np.isnan(row).any()
                else row
            )
            for row in valid_test_data
        ]
    )
    valid_test_data = valid_test_data.reshape(N, input_dim)

    # Create DataLoader for test data
    test_tensor = TensorDataset(torch.tensor(valid_test_data).float())
    test_loader = DataLoader(test_tensor, batch_size=2048, shuffle=False)

    # Load the pre-trained VAE model
    model = VAE(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Perform inference
    predictions_all = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            predictions, _, _ = model(batch)  # VAE model inference
            predictions_all.append(predictions.cpu().numpy())

    # Concatenate all batch predictions
    predictions_all = np.vstack(predictions_all)

    # Ensure predictions are in the correct shape
    if predictions_all.shape[-1] == 1:
        predictions_all = predictions_all.squeeze(-1)
    # if predictions_all.dim() == 3:
    # all_outputs = predictions_all.squeeze(1)

    # Initialize output array with NaNs
    outputs = np.full((test_data.shape[0], test_data.shape[1]), np.nan)

    # Fill in the valid mask positions with predictions
    outputs[mask] = predictions_all

    # Flatten latitude, longitude, and predictions for output
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()
    output_flat = outputs.flatten()

    # Combine latitude, longitude, and predictions
    final_output = np.column_stack((lat_flat, lon_flat, output_flat))

    # Save the final output including latitude and longitude
    if chla_data_file is None:
        chla_data_file = pace_filepath.replace(".nc", ".npy")
    np.save(chla_data_file, final_output)


def chla_viz(
    rgb_image_tif_file: str,
    chla_data_file: str,
    output_file: str,
    title: str = "PACE",
    figsize: tuple = (12, 8),
    cmap: str = "jet",
) -> None:
    """Visualizes the chlorophyll-a concentration over an RGB image.

    Args:
        rgb_image_tif_file (str): Path to the RGB image file.
        chla_data_file (str): Path to the chlorophyll-a data file.
        output_file (str): Path to save the output visualization.
        title (str, optional): Title of the plot. Defaults to "PACE".
        figsize (tuple, optional): Figure size for the plot. Defaults to (12, 8).
        cmap (str, optional): Colormap for the chlorophyll-a concentration. Defaults to "jet".
    """

    # Read RGB Image
    # rgb_image_tif_file = "data/snapshot-2024-08-10T00_00_00Z.tif"

    with rasterio.open(rgb_image_tif_file) as dataset:
        # Read R、G、B bands
        R = dataset.read(1)
        G = dataset.read(2)
        B = dataset.read(3)

        # # Get geographic extent, resolution information.
        extent = [
            dataset.bounds.left,
            dataset.bounds.right,
            dataset.bounds.bottom,
            dataset.bounds.top,
        ]
        transform = dataset.transform
        width, height = dataset.width, dataset.height

    # Combine the R, G, B bands into a 3D array.
    rgb_image = np.stack((R, G, B), axis=-1)

    # Load Chla data
    chla_data = np.load(chla_data_file)
    # chla_data = final_output

    # Extract the latitude, longitude, and concentration values of the chlorophyll-a data.
    latitude = chla_data[:, 0]
    longitude = chla_data[:, 1]
    chla_values = chla_data[:, 2]

    # Extract the pixels within the geographic extent of the RGB image.
    mask = (
        (latitude >= extent[2])
        & (latitude <= extent[3])
        & (longitude >= extent[0])
        & (longitude <= extent[1])
    )
    latitude = latitude[mask]
    longitude = longitude[mask]
    chla_values = chla_values[mask]

    # Create a grid with the same resolution as the RGB image.
    grid_lon = np.linspace(extent[0], extent[1], width)
    grid_lat = np.linspace(extent[3], extent[2], height)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Resample the chlorophyll-a data to the size of the RGB image using interpolation.
    chla_resampled = griddata(
        (longitude, latitude), chla_values, (grid_lon, grid_lat), method="linear"
    )

    # Keep NaN values as transparent regions.
    chla_resampled = np.ma.masked_invalid(chla_resampled)

    plt.figure(figsize=figsize)

    plt.imshow(rgb_image / 255.0, extent=extent, origin="upper")

    vmin, vmax = 0, 35
    im = plt.imshow(
        chla_resampled,
        extent=extent,
        cmap=cmap,
        alpha=0.6,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, orientation="horizontal")
    cbar.set_label("Chlorophyll-a Concentration (mg/m³)")

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # output_file = "20241024-2.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved overlay image to {output_file}")

    plt.show()


def npy_to_geotiff(
    npy_path,
    out_tif,
    resolution_m=1000,  # meters per pixel (lat exact, lon adjusted by cos(lat))
    method="linear",  # 'linear' | 'nearest' | 'cubic'
    nodata_val=-9999.0,
    bbox_padding=0.0,  # degrees padding around bbox
    lat_col=0,
    lon_col=1,
    band_cols=None,  # None=use all columns after lat/lon; or list like [2,3,5,...]
    band_names=None,  # Optional list of per-band descriptions, same length as output bands
    wavelengths=None,  # Optional list; if given, will be used in descriptions like "aphy_440"
    crs="EPSG:4326",
    compress="deflate",
    bigtiff="IF_SAFER",  # 'YES'|'NO'|'IF_NEEDED'|'IF_SAFER'
):
    """
    Convert [lat, lon, band1, band2, ...] scattered points in .npy into a multi-band GeoTIFF (EPSG:4326).

    Args:
        npy_path (str): Path to the .npy file.
        out_tif (str): Path to the output GeoTIFF file.
        resolution_m (int, optional): Resolution in meters per pixel. Defaults to 1000.
        method (str, optional): Method for interpolation. Defaults to "linear".
        nodata_val (float, optional): Value to fill NaN values. Defaults to -9999.0.
        bbox_padding (float, optional): Padding around the bounding box. Defaults to 0.0.
        lat_col (int, optional): Column index for latitude. Defaults to 0.
        lon_col (int, optional): Column index for longitude. Defaults to 1.
        band_cols (list, optional): Columns to rasterize as bands. Defaults to None.
        band_names (list, optional): Band descriptions. Defaults to None.
        wavelengths (list, optional): Wavelengths. Defaults to None.
        crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".
        compress (str, optional): Compression method. Defaults to "deflate".
        bigtiff (str, optional): BigTIFF mode. Defaults to "IF_SAFER".
    """

    # 1) Load & basic checks
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Must be 2D with >=3 columns.")

    lat = arr[:, lat_col].astype(float)
    lon = arr[:, lon_col].astype(float)

    # Decide which value columns become bands
    if band_cols is None:
        band_cols = [i for i in range(arr.shape[1]) if i not in (lat_col, lon_col)]
    if isinstance(band_cols, (int, np.integer)):
        band_cols = [int(band_cols)]
    if len(band_cols) == 0:
        raise ValueError("No value columns selected for bands.")

    # 2) Bounds (+ padding)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min -= bbox_padding
    lat_max += bbox_padding
    lon_min -= bbox_padding
    lon_max += bbox_padding

    # 3) Meter -> degree resolution (lat exact; lon scaled at center latitude)
    lat_center = (lat_min + lat_max) / 2.0
    deg_per_m_lat = 1.0 / 111000.0
    deg_per_m_lon = 1.0 / (111000.0 * np.cos(np.radians(lat_center)))
    res_lat_deg = resolution_m * deg_per_m_lat
    res_lon_deg = resolution_m * deg_per_m_lon

    # 4) Build grid (lon fastest axis -> width; lat -> height)
    lon_axis = np.arange(lon_min, lon_max + res_lon_deg, res_lon_deg)
    lat_axis = np.arange(lat_min, lat_max + res_lat_deg, res_lat_deg)
    Lon, Lat = np.meshgrid(lon_axis, lat_axis)

    # Precompute transform
    transform = from_origin(lon_axis.min(), lat_axis.max(), res_lon_deg, res_lat_deg)

    # 5) Interpolate each band onto the same grid
    grids = []
    for idx in band_cols:
        vals = arr[:, idx].astype(float)

        # Primary interpolation
        g = griddata(points=(lon, lat), values=vals, xi=(Lon, Lat), method=method)

        # Fill NaNs with nearest as a safety net
        if np.isnan(g).any():
            g_near = griddata(
                points=(lon, lat), values=vals, xi=(Lon, Lat), method="nearest"
            )
            g = np.where(np.isnan(g), g_near, g)

        # Flip vertically because raster origin is top-left
        grids.append(np.flipud(g).astype(np.float32))

    data_stack = np.stack(grids, axis=0)  # shape: (bands, height, width)

    # 6) Write GeoTIFF
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
        # Write bands
        for b in range(data_stack.shape[0]):
            band = data_stack[b]
            band[~np.isfinite(band)] = nodata_val
            dst.write(band, b + 1)

        # Add band descriptions
        descriptions = []
        n_bands = data_stack.shape[0]
        if band_names is not None and len(band_names) == n_bands:
            descriptions = list(map(str, band_names))
        elif wavelengths is not None and len(wavelengths) == n_bands:
            # e.g., "aphy_440"
            descriptions = [f"aphy_{int(wl)}" for wl in wavelengths]
        else:
            # Fallback: use column indices
            descriptions = [f"band_{band_cols[b]}" for b in range(n_bands)]

        for b in range(1, n_bands + 1):
            dst.set_band_description(b, descriptions[b - 1])

    # 7) Log
    print(f"✅ GeoTIFF：{out_tif}")
    print(f"   Size：{profile['width']} x {profile['height']} pixels")
    print(f"   Bands：{profile['count']}")
    print(
        f"   Resolution：{resolution_m} m/px (≈ {res_lon_deg:.6f}° × {res_lat_deg:.6f}° @ {lat_center:.2f}°)"
    )
    print(
        f"   Extent：lon[{lon_min:.6f}, {lon_max:.6f}], lat[{lat_min:.6f}, {lat_max:.6f}]"
    )
    print(f"   Descriptions：{descriptions}")
