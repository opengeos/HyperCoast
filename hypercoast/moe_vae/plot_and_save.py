import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import inspect
from .model import MoE_VAE, SparseDispatcher, VAE


def save_model_structure_from_classes(save_path):
    """Save the model structure from classes to a file.

    Args:
        save_path (str): Path to save the model structure.
    """
    classes = [SparseDispatcher, VAE, MoE_VAE]

    with open(save_path, "w") as f:
        for cls in classes:
            f.write(f"# === Source for {cls.__name__} ===\n")
            f.write(inspect.getsource(cls))
            f.write("\n\n")


def calculate_metrics(predictions, actuals, threshold=0.8):
    """Calculate epsilon, beta and additional metrics (RMSE, RMSLE, MAPE, Bias, MAE).

    Args:
        predictions (array-like): Predicted values.
        actuals (array-like): Actual values.
        threshold (float, optional): Relative error threshold. Defaults to 0.8.

    Returns:
        tuple: Tuple containing:
            - epsilon (float): Epsilon value.
            - beta (float): Beta value.
            - rmse (float): Root mean square error.
            - rmsle (float): Root mean square log error.
            - mape (float): Mean absolute percentage error.
            - bias (float): Bias value.
            - mae (float): Mean absolute error.
    """
    # Apply the threshold to filter out predictions with large relative error
    # mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < threshold
    # filtered_predictions = predictions[mask]
    # filtered_actuals = actuals[mask]
    predictions = np.where(predictions <= 1e-10, 1e-10, predictions)
    actuals = np.where(actuals <= 1e-10, 1e-10, actuals)
    filtered_predictions = predictions
    filtered_actuals = actuals

    # Calculate epsilon and beta
    log_ratios = np.log10(filtered_predictions / filtered_actuals)
    Y = np.median(np.abs(log_ratios))
    Z = np.median(log_ratios)
    epsilon = 100 * (10**Y - 1)
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
    predictions_rescaled, actuals_rescaled, save_dir, threshold=10, mode="test"
):
    """Plot the results of the MoE-VAE model.

    Args:
        predictions_rescaled (array-like): Predicted values.
        actuals_rescaled (array-like): Actual values.
        save_dir (str): Directory to save the plot.
        threshold (float, optional): Relative error threshold. Defaults to 10.
        mode (str, optional): Mode of the plot. Defaults to "test".
    """

    actuals = actuals_rescaled.flatten()
    predictions = predictions_rescaled.flatten()

    log_actuals = np.log10(actuals)
    log_predictions = np.log10(predictions)

    # mask = np.abs(predictions - actuals) / np.abs(actuals+1e-10) < threshold
    mask = np.abs(log_predictions - log_actuals) < threshold
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
    x = np.array([-4, 4])
    y = slope * x + intercept

    plt.figure(figsize=(6, 6))

    plt.plot(x, y, linestyle="--", color="blue", linewidth=0.8)
    lims = [-4, 4]
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

    plt.xlabel("Actual Values", fontsize=16, fontname="Ubuntu")
    plt.ylabel("Predicted Values", fontsize=16, fontname="Ubuntu")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, which="both", ls="--")

    plt.legend(
        title=(
            f"MAE = {mae:.2f}, RMSE = {rmse:.2f}, RMSLE = {rmsle:.2f} \n"
            f"Bias = {bias:.2f}, Slope = {slope:.2f} \n"
            f"MAPE = {mape:.2f}%, ε = {epsilon:.2f}%, β = {beta:.2f}%"
        ),
        fontsize=16,
        title_fontsize=12,
        prop={"family": "Ubuntu"},
    )

    plt.xticks(fontsize=20, fontname="Ubuntu")
    plt.yticks(fontsize=20, fontname="Ubuntu")

    plt.savefig(os.path.join(save_dir, f"{mode}_plot.pdf"), bbox_inches="tight")
    plt.close()


def save_results_to_excel(ids, actuals, predictions, file_path, dates=None):
    """
    Save prediction results to an Excel file.
    - If dates are included, output ID, Date, Actual, Predicted;
    - Otherwise, output ID, Actual, Predicted.

    Args:
        ids (array-like): IDs of the samples.
        actuals (array-like): Actual values.
        predictions (array-like): Predicted values.
        file_path (str): Path to save the Excel file.
        dates (array-like, optional): Dates of the samples. Defaults to None.
    """
    if dates is not None:
        df = pd.DataFrame(
            {"ID": ids, "Date": dates, "Actual": actuals, "Predicted": predictions}
        )
    else:
        df = pd.DataFrame({"ID": ids, "Actual": actuals, "Predicted": predictions})

    df.to_excel(file_path, index=False)


def save_and_plot_results_from_excel(
    predictions, actuals, sample_ids, dates, original_excel_path, save_dir
):
    """Save and plot the results from an Excel file.

    Args:
        predictions (array-like): Predicted values.
        actuals (array-like): Actual values.
        sample_ids (array-like): IDs of the samples.
        dates (array-like): Dates of the samples.
        original_excel_path (str): Path to the original Excel file.
        save_dir (str): Directory to save the results.
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
    plot_results(predictions, actuals, save_dir, mode=dataset_name)
