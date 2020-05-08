import gc
import numpy as np
import matplotlib.pyplot as plt
from ml_editor import constants
from ml_editor.utils import format_file_name

plt.style.use(constants.MATPLOTLIB_STYLE)


def save_lin_reg_plot(
    manufacturer: str,
    model: str,
    year: int,
    X: np.array,
    y: np.array,
    y_preds: np.array,
) -> None:
    """
    Saves a scatter plot of the data, linear regression line and 
    boudaries for the different deal levels
    :param manufacturer: manufacturer of the vechicle
    :param model: model of the vehicle
    :param year: year of manufacture
    :param X: array of odometer values for the plot
    :param y: array of price values for the plot
    :param y_preds: predicitons for the odometer values from the regression
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_title(f"{manufacturer}, {model} from {year}")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")
    ax.scatter(X, y)
    ax.plot(np.sort(X, axis=0), y_preds)

    # Plotting the confidence regions which are are determined
    # by the mean absolute residual of the regression
    ax.plot(
        np.sort(X, axis=0),
        y_preds - constants.RESIDUAL_FACTOR_DIFF * np.mean(abs(y_preds - y)),
        label="Good deal",
        ls="--",
        lw=0.7,
        c="green",
    )
    ax.plot(
        np.sort(X, axis=0),
        y_preds - 2 * constants.RESIDUAL_FACTOR_DIFF * np.mean(abs(y_preds - y)),
        label="Very good deal",
        ls="--",
        lw=0.7,
        c="darkgreen",
    )
    ax.plot(
        np.sort(X, axis=0),
        y_preds + constants.RESIDUAL_FACTOR_DIFF * np.mean(abs(y_preds - y)),
        label="Bad deal",
        ls="--",
        lw=0.7,
        c="red",
    )
    ax.plot(
        np.sort(X, axis=0),
        y_preds + 2 * constants.RESIDUAL_FACTOR_DIFF * np.mean(abs(y_preds - y)),
        label="Very bad deal",
        ls="--",
        lw=0.7,
        c="darkred",
    )
    ax.legend()
    file_name = f"../models/images/{manufacturer}_{model}_{year}.pkl"
    fig.savefig(format_file_name(file_name), bbox_inches="tight")

    # Attempt to properly close the figure and garbage collect as memory usage accumulating
    # as all the models are trained
    plt.close("all")
    gc.collect()
