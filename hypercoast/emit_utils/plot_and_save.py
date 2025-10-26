"""Plotting and result saving utilities for model evaluation.

This module provides functions for visualizing model predictions, computing
performance metrics, and saving results to Excel files.
"""

from typing import List, Optional, Tuple

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter
    from scipy.stats import gaussian_kde
except ImportError:
    pass


def calculate_metrics(
    predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.8
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate performance metrics for water quality parameter predictions.

    Computes various metrics including epsilon, beta (from IOCCG protocols),
    NRMSE, RMSLE, MAPE, bias, and MAE for evaluating model performance.

    Args:
        predictions: Array of predicted values.
        actuals: Array of actual (ground truth) values.
        threshold: Relative error threshold (not currently used in filtering).

    Returns:
        epsilon: Symmetric signed percentage difference (IOCCG).
        beta: Bias percentage (IOCCG).
        nrmse: Normalized root mean squared error.
        rmsle: Root mean squared logarithmic error.
        mape: Median absolute percentage error.
        bias: Multiplicative bias.
        mae: Median absolute error in log space (antilog).
    """
    eps = 1e-10  # small constant to avoid division by zero

    predictions = np.where(predictions <= eps, eps, predictions)
    actuals = np.where(actuals <= eps, eps, actuals)
    filtered_predictions = predictions
    filtered_actuals = actuals

    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / filtered_actuals)
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
    beta = 50 * np.sign(Z) * (10 ** np.abs(Z) - 1)

    # NRMSE: RMSE normalized by range (max - min)
    rmse = np.sqrt(np.mean((filtered_predictions - filtered_actuals) ** 2))
    nrmse = rmse / (np.max(filtered_actuals) - np.min(filtered_actuals) + eps)

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

    return epsilon, beta, nrmse, rmsle, mape, bias, mae


def plot_results(
    predictions_rescaled: np.ndarray,
    actuals_rescaled: np.ndarray,
    save_dir: str,
    threshold: float = 10,
    mode: str = "test",
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
) -> None:
    """Create scatter plot with KDE contours comparing predictions vs actuals.

    Generates a log-log scatter plot with regression line, 1:1 reference line,
    KDE density contours, and performance metrics in the legend.

    Args:
        predictions_rescaled: Array of predicted values.
        actuals_rescaled: Array of actual (ground truth) values.
        save_dir: Directory to save the output plot.
        threshold: Threshold for filtering outliers in log space.
        mode: Name prefix for the output file (e.g., 'test', 'train').
        xlim: X-axis limits in log space (e.g., (-4, 4) for 10^-4 to 10^4).
        ylim: Y-axis limits in log space.
    """
    os.makedirs(save_dir, exist_ok=True)

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    log_actuals = np.log10(np.where(actuals == 0, 1e-10, actuals))
    log_predictions = np.log10(np.where(predictions == 0, 1e-10, predictions))

    mask = np.abs(log_predictions - log_actuals) < threshold
    filtered_predictions = predictions[mask]
    filtered_actuals = actuals[mask]

    filtered_log_actual = np.log10(
        np.where(filtered_actuals == 0, 1e-10, filtered_actuals)
    )
    filtered_log_prediction = np.log10(
        np.where(filtered_predictions == 0, 1e-10, filtered_predictions)
    )

    epsilon, beta, nrmse, rmsle, mape, bias, mae = calculate_metrics(
        filtered_predictions, filtered_actuals, threshold
    )

    valid_mask = np.isfinite(filtered_log_actual) & np.isfinite(filtered_log_prediction)
    slope, intercept = np.polyfit(
        filtered_log_actual[valid_mask], filtered_log_prediction[valid_mask], 1
    )
    x = np.array([xlim[0], xlim[1]])
    y = slope * x + intercept

    plt.figure(figsize=(6, 6))

    # Regression line
    plt.plot(x, y, linestyle="--", color="blue", linewidth=0.8)
    # 1:1 line
    plt.plot(xlim, ylim, linestyle="-", color="black", linewidth=0.8)

    # Scatter & KDE
    sns.scatterplot(x=log_actuals, y=log_predictions, alpha=0.5)
    sns.kdeplot(
        x=filtered_log_actual,
        y=filtered_log_prediction,
        levels=3,
        color="black",
        fill=False,
        linewidths=0.8,
    )

    plt.xlabel("Actual Values", fontsize=16, fontname="Ubuntu")
    plt.ylabel("Predicted Values", fontsize=16, fontname="Ubuntu")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, which="both", ls="--")

    plt.legend(
        title=(
            f"MAE = {mae:.2f}, NRMSE = {nrmse:.2f}, RMSLE = {rmsle:.2f}\n"
            f"Bias = {bias:.2f}, Slope = {slope:.2f}\n"
            f"MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=12,
        title_fontsize=10,
        prop={"family": "Ubuntu"},
    )

    plt.xticks(fontsize=14, fontname="Ubuntu")
    plt.yticks(fontsize=14, fontname="Ubuntu")

    png_path = os.path.join(save_dir, f"{mode}_plot.png")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"✅ Saved and displayed: {png_path}")


def plot_results_with_density(
    predictions_rescaled: np.ndarray,
    actuals_rescaled: np.ndarray,
    save_dir: str,
    threshold: float = 10,
    mode: str = "test_density",
    xlim: Tuple[float, float] = (-4, 4),
    ylim: Tuple[float, float] = (-4, 4),
    cmap: str = "viridis",
    tick_min: float = -4,
    tick_max: float = 4,
    tick_step: float = 1,
) -> None:
    """Create density-colored scatter plot comparing predictions vs actuals.

    Generates a log-log scatter plot where points are colored by local density
    estimated using Gaussian KDE. Includes regression line, 1:1 reference line,
    and performance metrics.

    Args:
        predictions_rescaled: Array of predicted values.
        actuals_rescaled: Array of actual (ground truth) values.
        save_dir: Directory to save the output plot.
        threshold: Threshold for filtering outliers in log space.
        mode: Name prefix for the output file.
        xlim: X-axis limits in log space.
        ylim: Y-axis limits in log space.
        cmap: Colormap name for density visualization.
        tick_min: Minimum tick value in log space.
        tick_max: Maximum tick value in log space.
        tick_step: Step between ticks in log space.
    """
    os.makedirs(save_dir, exist_ok=True)

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    # Log10 transform (avoid non-positive values)
    eps = 1e-10
    log_actuals = np.log10(np.where(actuals <= 0, eps, actuals))
    log_predictions = np.log10(np.where(predictions <= 0, eps, predictions))

    # Optional threshold filtering
    mask = np.abs(log_predictions - log_actuals) < threshold
    log_a_f, log_p_f = log_actuals[mask], log_predictions[mask]
    a_f, p_f = actuals[mask], predictions[mask]

    # Density estimation with Gaussian KDE
    xy = np.vstack([log_a_f, log_p_f])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    log_a_f, log_p_f, z = log_a_f[idx], log_p_f[idx], z[idx]
    a_f, p_f = a_f[idx], p_f[idx]

    # Calculate metrics
    epsilon, beta, nrmse, rmsle, mape, bias, mae = calculate_metrics(
        p_f, a_f, threshold
    )

    # Linear regression (in log-log space)
    valid_mask = np.isfinite(log_a_f) & np.isfinite(log_p_f)
    slope, intercept = np.polyfit(log_a_f[valid_mask], log_p_f[valid_mask], 1)

    # === Plot ===
    plt.figure(figsize=(8, 6), dpi=300)

    # 1:1 reference line (do not add to legend)
    plt.plot(
        [xlim[0], xlim[1]],
        [xlim[0], xlim[1]],
        linestyle="-",
        color="black",
        linewidth=0.9,
    )

    # Regression line (do not add to legend)
    xs = np.array([xlim[0], xlim[1]])
    ys = slope * xs + intercept
    plt.plot(xs, ys, linestyle="--", color="blue", linewidth=0.9)

    # Scatter points colored by density
    sc = plt.scatter(log_a_f, log_p_f, c=z, s=30, cmap=cmap, alpha=1, edgecolors="none")

    # Colorbar for density
    cbar = plt.colorbar(sc, fraction=0.06, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    # Axis ticks (shown as powers of 10)
    ax = plt.gca()
    ticks = np.arange(tick_min, tick_max + 1e-9, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    formatter = FuncFormatter(lambda val, pos: f"$10^{{{val:.1f}}}$")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=16)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, ls="--", alpha=0.5)

    # Metrics in legend (only title shown)
    plt.legend(
        title=(
            f"MAE = {mae:.2f}, NRMSE = {nrmse:.2f}\n"
            f"RMSLE = {rmsle:.2f}, Bias = {bias:.2f}\n"
            f"MAPE = {mape:.2f}%, Slope = {slope:.2f}\n"
            f"ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=12,
        title_fontsize=10,
        frameon=True,
    )

    # Save figure as PNG
    png_path = os.path.join(save_dir, f"{mode}.png")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.show()

    print(f"✅ Saved and displayed PNG: {png_path}")


def save_results_to_excel(
    ids: List[str],
    actuals: np.ndarray,
    predictions: np.ndarray,
    file_path: str,
    dates: Optional[List[str]] = None,
) -> None:
    """Save prediction results to an Excel file.

    Args:
        ids: List of sample identifiers.
        actuals: Array of actual (ground truth) values.
        predictions: Array of predicted values.
        file_path: Output path for the Excel file.
        dates: Optional list of date strings to include in output.
    """
    if dates is not None:
        df = pd.DataFrame(
            {"ID": ids, "Date": dates, "Actual": actuals, "Predicted": predictions}
        )
    else:
        df = pd.DataFrame({"ID": ids, "Actual": actuals, "Predicted": predictions})

    df.to_excel(file_path, index=False)


def save_results_from_excel_for_test(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sample_ids: List[str],
    dates: List[str],
    original_excel_path: str,
    save_dir: str,
) -> None:
    """Save test results to Excel file with dataset name from original file.

    Args:
        predictions: Array of predicted values.
        actuals: Array of actual (ground truth) values.
        sample_ids: List of sample identifiers.
        dates: List of date strings.
        original_excel_path: Path to original Excel file (used for naming output).
        save_dir: Directory to save the output Excel file.
    """
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(original_excel_path)
    dataset_name = os.path.splitext(filename)[0]

    save_results_to_excel(
        sample_ids,
        actuals,
        predictions,
        os.path.join(save_dir, f"{dataset_name}.xlsx"),
        dates=dates,
    )
