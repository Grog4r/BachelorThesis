import colorsys
import os
from datetime import datetime
from turtle import width

import pandas as pd
import plotly.graph_objects as go

if __name__ == "__main__" or os.environ.get("EXPERIMENT") == "True":
    from train_regression_model import find_divergence_time
else:
    from utilities.train_regression_model import find_divergence_time


def string_to_color(input_string: str) -> str:
    """Hashes a string to a color.
        This function will also ensure that the color is not too light.

    :param input_string: The string to hash.
    :return: The color value for the input string.
    """
    # Use the hash function to generate a unique integer for the input string
    hash_code = hash(input_string)

    # Convert the hash code to a 6-digit hexadecimal color code
    color_code = f"#{(hash_code & 0xFFFFFF):06x}"

    # Convert the color code to RGB values
    r, g, b = (
        int(color_code[1:3], 16),
        int(color_code[3:5], 16),
        int(color_code[5:7], 16),
    )

    # Calculate the brightness of the color
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    # If the color is too light, darken it
    if brightness > 200:
        r = max(int(r * 0.8), 50)
        g = max(int(g * 0.8), 50)
        b = max(int(b * 0.8), 50)

    # Convert the RGB values back to a color code
    color_code = f"#{r:02x}{g:02x}{b:02x}"

    return color_code


def generate_brightness_shades(color_code: str, n: int) -> list[str]:
    """Generates a list of brightness values for a given color code

    :param color_code: The base color code
    :param n: The number of brightness shades to generate
    :return: The list of color shade codes
    """
    r, g, b = tuple(int(color_code[i : i + 2], 16) for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

    brightness_shades = []
    for i in range(n):
        new_v = v * (1 - i * (1 / n))
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)
        new_color_code = (
            f"#{int(new_r * 255):02x}{int(new_g * 255):02x}{int(new_b * 255):02x}"
        )
        brightness_shades.append(new_color_code)

    return brightness_shades


def plot_device(
    df: pd.DataFrame,
    device_uuid: str,
    col_to_plot: str = "battery_level_percent",
    size: tuple[int, int] | None = (1000, 600),
    color: str | None = None,
    title: str | None = None,
    y_desc: str | None = None,
    fixed_y_axis: bool = True,
) -> None:
    """Plots a column for a given device.

    :param df: The DataFrame to plot.
    :param device_uuid: The device to plot.
    :param col_to_plot: The column to plot., defaults to "battery_level_percent"
    :param size: The size of the plot (width, height), if this is None the plot will be sized automatically,
        defaults to (1000, 600)
    :param color: The color of the curve, if this is None the color will be chosen automatically,
        defaults to None
    :param title: The title of the plot, if this is None the title will be chosen automatically,
        defaults to None
    :param fixed_y_axis: Wheter or not to use a fixed y axis, defaults to True
    """
    fig = go.Figure()

    device_df = df[df["device_uuid"] == device_uuid]

    fig.add_trace(
        go.Scatter(
            x=device_df["status_time"],
            y=device_df[col_to_plot],
            mode="lines",
            name=device_uuid[-8:],
            marker=dict(color=string_to_color(device_uuid) if color is None else color),
        )
    )

    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    fig.update_layout(
        xaxis=dict(title="Zeit"),
        yaxis=(dict(title="Batterieladung in %" if y_desc is None else y_desc)),
        title=f"Batterieladung für Gerät {device_uuid}" if title is None else title,
        showlegend=True,
        **size,
    )
    if fixed_y_axis:
        fig.update_yaxes(range=[0, 100])

    fig.show()


def plot_devices(
    df: pd.DataFrame,
    col_to_plot: str = "battery_level_percent",
    size: tuple[int, int] | None = (1000, 600),
    color: str | None = None,
    title: str | None = None,
    y_desc: str | None = None,
    fixed_y_axis: bool = True,
) -> None:
    """Plots all the devices for a DataFrame.

    :param df: The DataFrame to plot.
    :param col_to_plot: The column to plot, defaults to "battery_level_percent"
    :param size: The size of the plot (width, height), if this is None the plot will be sized automatically,
        defaults to (1000, 600)
    :param color: The color of the curve, if this is None the color will be chosen automatically,
        defaults to None
    :param title: The title of the plot, if this is None the title will be chosen automatically,
        defaults to None
    :param fixed_y_axis: Wheter or not to use a fixed y axis, defaults to True
    """
    fig = go.Figure()

    for device, group in df.groupby("device_uuid"):
        fig.add_trace(
            go.Scatter(
                x=group["status_time"],
                y=group[col_to_plot],
                mode="lines",
                name=device[-8:],
                marker=dict(color=string_to_color(device) if color is None else color),
            )
        )

    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    fig.update_layout(
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Batterieladung in %" if y_desc is None else y_desc),
        title="Batterieladung für Geräte" if title is None else title,
        showlegend=True,
        **size,
    )
    if fixed_y_axis:
        fig.update_yaxes(range=[0, 100])

    fig.show()


def plot_devices_and_peaks(
    df: pd.DataFrame,
    col_to_plot: str = "battery_level_percent",
    peak_col: str = "peak_label",
    plot_peak_borders: bool = False,
    size: tuple[int, int] | None = (1000, 600),
    color: str | None = None,
    title: str | None = None,
) -> None:
    """Plots all devices and their peaks in a given DataFrame.

    :param df: The DataFrame to plot.
    :param col_to_plot: The column to plot, defaults to "battery_level_percent"
    :param peak_col: The column where the peaks are saved in.
    :param plot_peak_borders: If the left and right border of the peak should also be
        plotted, defaults to False
    :param size: The size of the plot (width, height), if this is None the plot will be sized automatically,
        defaults to (1000, 600)
    :param color: The color of the curve, if this is None the color will be chosen automatically,
        defaults to None
    :param title: The title of the plot, if this is None the title will be chosen automatically,
        defaults to None
    """
    fig = go.Figure()

    for device, group in df.groupby("device_uuid"):
        fig.add_trace(
            go.Scatter(
                x=group["status_time"],
                y=group[col_to_plot],
                mode="lines",
                legendgroup=device,
                name=device[-8:],
                marker=dict(color=string_to_color(device) if color is None else color),
            )
        )

        # Adding vertical lines at peak points
        peaks = group[group[peak_col] != 0.0]
        for _, peak in peaks.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[peak["status_time"]] * 2,
                    y=[0, 100],
                    mode="lines",
                    line=dict(
                        color="red",
                        width=2,
                        dash="dashdot",
                    ),
                    legendgroup=device,
                    showlegend=False,
                )
            )

        if plot_peak_borders:
            # Adding vertical lines left of peak points
            peaks_left = group[group["left_peak_border"]]
            for _, peak in peaks_left.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[peak["status_time"]] * 2,
                        y=[0, 100],
                        mode="lines",
                        line=dict(
                            color="blue",
                            width=2,
                            dash="dashdot",
                        ),
                        legendgroup=device,
                        showlegend=False,
                    )
                )

            # Adding vertical lines right of peak points
            peaks_right = group[group["right_peak_border"]]
            for _, peak in peaks_right.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[peak["status_time"]] * 2,
                        y=[0, 100],
                        mode="lines",
                        line=dict(
                            color="blue",
                            width=2,
                            dash="dashdot",
                        ),
                        legendgroup=device,
                        showlegend=False,
                    )
                )

    fig.update_yaxes(range=[0, 100])

    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    fig.update_layout(
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Batterieladung in %"),
        title=(
            "Batterieladungfür die Geräte mit eingezeichneten Zyklusgrenzen"
            if title is None
            else title
        ),
        showlegend=True,
        **size,
    )

    fig.show()


def plot_compare_multiple_dfs(
    to_plot: list[tuple[pd.DataFrame, str, str, str, bool]],
    fixed_y_axis: bool = True,
    group_by_cycle: bool = False,
    colors: list[str] | None = None,
    size: tuple[int, int] | None = (1000, 600),
    title: str | None = None,
    y_desc: str | None = None,
) -> None:
    """Plots multiple DataFrames together in a single plot.

    :param to_plot: A list of tuples with the data to plot:
        df: The DataFrame to plot from
        col_to_plot: The column in the DataFrame to plot together
        device_to_plot: The device in the DataFrame to plot together
        name: The name for the curve
        contains: If True the device_uuid only needs to contain the device_to_plot,
            else it has to be exactly the same.
    :param fixed_y_axis: If True the y axis will always go from 0 to 100,
        defaults to False
    :param group_by_cycle: Plot the cycles separately if True, defaults to False
    :param colors: If this is None the colors will be chosen randomly, else the colors
        in the list will be used, defaults to None
    :param y_desc: The descripion of the y-Axis, if this is None it will be automatically be set,
        defaults to None
    """
    fig = go.Figure()

    for i, element in enumerate(to_plot):
        (df, col_to_plot, device_uuid, name, contains) = element
        if colors == None:
            color = string_to_color(device_uuid + name)
        else:
            color = colors[i]

        if contains:
            device_df = df[df["device_uuid"].str.contains(device_uuid)]
        else:
            device_df = df[df["device_uuid"] == device_uuid]
        if group_by_cycle:
            for cycle_id, cycle_group in df.groupby("cycle_id"):
                fig.add_trace(
                    go.Scatter(
                        x=cycle_group["status_time"],
                        y=cycle_group[col_to_plot],
                        mode="lines",
                        name=f"{name} Cycle {cycle_id}",
                        marker=dict(color=color),
                        legendgroup=cycle_id,
                        legendgrouptitle=dict(text=f"Cycle {cycle_id}"),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=device_df["status_time"],
                    y=device_df[col_to_plot],
                    mode="lines",
                    name=f"{name}",
                    marker=dict(color=color),
                )
            )

    if fixed_y_axis:
        fig.update_yaxes(range=[0, 100])

    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    fig.update_layout(
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Batterieladung in %" if y_desc is None else y_desc),
        title="Batterieladung für Geräte" if title is None else title,
        showlegend=True,
        **size,
    )

    fig.show()


def plot_iterative_prediction(
    test_df: pd.DataFrame,
    pred_dfs: list[tuple[pd.DataFrame, str]],
    prediction_start_date: datetime,
    n_dev: int = -1,
    col_to_plot: str = "battery_level_percent",
    divergence_threshold: float = 100,
    title: str | None = None,
    size: tuple[int, int] | None = (1000, 600),
) -> go.Figure:
    """Plots a test dataset against iterative predictions

    :param real_df: The test dataset
    :param pred_dfs: A list of tuples containing the predictions and their names
    :param prediction_start_date: The start date of the prediction
    :param n_dev: The number of devices the model was trained on, will be ignored if this is -1, defaults to -1
    :param col_to_plot: The column to plot, defaults to "battery_level_percent"
    :param divergence_threshold: The threshold for the divergence line, defaults to 100
    :param title: The title of the plot, if this is None the title will be chosen automatically,
        defaults to None
    :param size: The size of the plot (width, height), if this is None the plot will be sized automatically,
        defaults to (1000, 600)
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_df["status_time"],
            y=test_df[col_to_plot],
            mode="lines",
            name="Target",
            marker=dict(color="red"),
        )
    )

    colors = generate_brightness_shades("#0000ff", n=len(pred_dfs))
    for i, ((pred_df, name), color) in enumerate(zip(pred_dfs, colors)):
        fig.add_trace(
            go.Scatter(
                x=pred_df["status_time"],
                y=pred_df[col_to_plot],
                mode="lines",
                name=f"Vorhersage",
                marker=dict(color=color),
                legendgroup=i,
                legendgrouptitle=dict(text=f"Vorhersage {name}"),
            )
        )

        divergence_time = find_divergence_time(
            test_df, pred_df, divergence_threshold, col=col_to_plot
        )
        if divergence_time is not None:
            print("Adding divergence line")
            fig.add_trace(
                go.Scatter(
                    x=[divergence_time] * 2,
                    y=[0, 100],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2,
                        dash="dashdot",
                    ),
                    name=f"Erste Abweichungs von {divergence_threshold}%",
                    legendgroup=i,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[prediction_start_date] * 2,
            y=[0, 100],
            mode="lines",
            line=dict(
                color="green",
                width=2,
                dash="dashdot",
            ),
            name="Beginn der Vorhersage",
        )
    )

    fig.update_yaxes(range=[0, 100])
    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    if title is None:
        title = (
            f"Iterative Vorhersage für die besten Modelle mit {n_dev} Trainingsgeräten."
            if n_dev != -1
            else ""
        )
    fig.update_layout(
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Batterieladung in %"),
        title=title,
        showlegend=True,
        autosize=False,
        **size,
    )

    fig.show()
    return fig


def plot_cycles(
    df: pd.DataFrame,
    x_is_index: bool = False,
    grouped_by_device: bool = False,
    size: tuple[int, int] | None = (1000, 600),
    title: str | None = None,
) -> None:
    """Plots the cycles of a dataset.

    :param df: The dataset where the cycles are in. Cycles are defined by the column
        cycle_id. The dataset is expected to come sorted by the device_uuids and the
        status_time.
    :param x_is_index: Defines if the x-axis should be in time or in the index of the
        values, defaults to False
    :param grouped_by_device: _description_, Defines if the cycles should be grouped by
        their device_uuid, defaults to False
    :param size: The size of the plot (width, height), if this is None the plot will be sized automatically,
        defaults to (1000, 600)
    :param title: The title of the plot, if this is None the title will be chosen automatically,
        defaults to None
    """
    # Create a figure
    fig = go.Figure()

    if not grouped_by_device:
        # Plot cycles grouped by their cycle_id
        for cycle_id, cycle_group in df.groupby("cycle_id"):
            fig.add_trace(
                go.Scatter(
                    x=(
                        list(range(len(cycle_group)))
                        if x_is_index
                        else cycle_group["status_time"]
                    ),
                    y=cycle_group["battery_level_percent"],
                    mode="lines",
                    name=f"Cycle {int(cycle_id)}",
                )
            )
    else:
        # Plot cycles grouped by their device_uuid
        for device_uuid, device_group in df.groupby("device_uuid"):
            device_color = string_to_color(device_uuid)
            n_cycles = len(device_group["cycle_id"].unique())
            color_shades = generate_brightness_shades(device_color, n_cycles)
            i = 0
            for cycle_id, cycle_group in device_group.groupby("cycle_id"):
                fig.add_trace(
                    go.Scatter(
                        x=(
                            list(range(len(cycle_group)))
                            if x_is_index
                            else cycle_group["status_time"]
                        ),
                        y=cycle_group["battery_level_percent"],
                        mode="lines",
                        marker=dict(color=color_shades[i]),
                        name=f"Cycle {int(cycle_id)}",
                        legendgroup=device_uuid,
                        legendgrouptitle=dict(text=device_uuid[-8:]),
                        showlegend=True,
                    )
                )
                i += 1

    # Update layout
    fig.update_yaxes(range=[0, 100])

    if size is not None:
        size = {"width": size[0], "height": size[1]}
    else:
        size = {}
    fig.update_layout(
        xaxis=dict(title="Index (Reset for Each Cycle)" if x_is_index else "Date"),
        yaxis=dict(title="Batterieladung in %"),
        title="Discharge Cycles" if title is None else title,
        showlegend=True,
        **size,
    )

    # Show the plot
    fig.show()
