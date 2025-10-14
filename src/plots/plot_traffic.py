import matplotlib.pyplot as plt
import numpy as np
import os

import constants
from config_classes import model_file_paths
from functions.average_data_func import average_data_func
from .shared_plot_data import shared_plot_data
from .helpers import (
    matlab_datenum_to_datetime_array,
    format_time_axis,
    mask_nodata,
    generate_plot_filename,
)


def plot_traffic(shared: shared_plot_data, paths: model_file_paths) -> None:
    """
    Plot traffic-related figure (plot_figure 1 in MATLAB):
    - Traffic volume (total, light, heavy, studded light, winter light)
    - Traffic speed (light, heavy)
    - Salting and sanding (stairs)
    """

    # Shorthands
    date_num = shared.date_num
    n_date = shared.date_num.shape[0]
    nodata = shared.nodata
    i_min, i_max = shared.i_min, shared.i_max
    av = list(shared.av)

    # Local masked copies
    traffic = mask_nodata(shared.traffic_data_ro.copy(), nodata)
    activity = mask_nodata(shared.activity_data_ro.copy(), nodata)

    # Prepare series for panel 1: traffic volume
    x_str, xplot, y_total = average_data_func(
        date_num, traffic[constants.N_total_index, :n_date], i_min, i_max, av
    )
    _, _, y_li = average_data_func(
        date_num, traffic[constants.N_v_index[constants.li], :n_date], i_min, i_max, av
    )
    _, _, y_he = average_data_func(
        date_num, traffic[constants.N_v_index[constants.he], :n_date], i_min, i_max, av
    )
    _, _, y_studded_li = average_data_func(
        date_num,
        traffic[constants.N_t_v_index[(constants.st, constants.li)], :n_date],
        i_min,
        i_max,
        av,
    )
    _, _, y_winter_li = average_data_func(
        date_num,
        traffic[constants.N_t_v_index[(constants.wi, constants.li)], :n_date],
        i_min,
        i_max,
        av,
    )
    dt_x = matlab_datenum_to_datetime_array(xplot)

    # Prepare panel 2: traffic speed
    _, _, y_v_li = average_data_func(
        date_num,
        traffic[constants.V_veh_index[constants.li], :n_date],
        i_min,
        i_max,
        av,
    )
    _, _, y_v_he = average_data_func(
        date_num,
        traffic[constants.V_veh_index[constants.he], :n_date],
        i_min,
        i_max,
        av,
    )

    # Prepare panel 3: salting/sanding stairs
    _, _, y_sanding = average_data_func(
        date_num,
        activity[constants.M_sanding_index, :n_date] / 10.0,
        i_min,
        i_max,
        av,
    )
    _, _, y_salting_na = average_data_func(
        date_num,
        activity[constants.M_salting_index[0], :n_date],
        i_min,
        i_max,
        av,
    )
    _, _, y_salting_2 = average_data_func(
        date_num,
        activity[constants.M_salting_index[1], :n_date],
        i_min,
        i_max,
        av,
    )

    # Create figure with three stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # Set window title for the figure window
    try:
        fig.canvas.manager.set_window_title("Figure 1: Traffic")  # type: ignore
    except Exception:
        pass

    # Panel 1: traffic volume
    ax1.set_title(
        f"{paths.title_str}: Traffic" if getattr(paths, "title_str", "") else "Traffic"
    )
    ax1.plot(dt_x, y_total.squeeze(), "k-", linewidth=1, label="Total")
    ax1.plot(dt_x, y_li.squeeze(), "b--", linewidth=0.8, label="Light")
    ax1.plot(dt_x, y_he.squeeze(), "r--", linewidth=0.8, label="Heavy")
    ax1.plot(dt_x, y_studded_li.squeeze(), "m:", linewidth=0.8, label="Light studded")
    ax1.plot(dt_x, y_winter_li.squeeze(), "g--", linewidth=0.8, label="Light winter")
    ax1.set_ylabel("Traffic volume (veh/hr)")
    ax1.legend(loc="upper left")
    format_time_axis(ax1, dt_x, shared.av[0], day_tick_limit=150)
    if len(dt_x) > 0:
        dt_min = matlab_datenum_to_datetime_array(np.array([date_num[i_min]]))[0]
        dt_max = matlab_datenum_to_datetime_array(np.array([date_num[i_max]]))[0]
        ax1.set_xlim(dt_min, dt_max)

    # Panel 2: traffic speed
    ax2.plot(dt_x, y_v_li.squeeze(), "b--", linewidth=1, label="Light")
    ax2.plot(dt_x, y_v_he.squeeze(), "r--", linewidth=0.8, label="Heavy")
    ax2.set_ylabel("Traffic speed (km/hr)")
    ax2.legend(loc="upper left")
    format_time_axis(ax2, dt_x, shared.av[0], day_tick_limit=150)

    # Panel 3: salting/sanding (stairs)
    ax3.step(
        dt_x,
        y_sanding.squeeze(),
        where="post",
        color="b",
        linewidth=1,
        label="Sanding/10",
    )
    ax3.step(
        dt_x,
        y_salting_na.squeeze(),
        where="post",
        color="g",
        linewidth=1,
        label="Salting(na)",
    )
    ax3.step(
        dt_x,
        y_salting_2.squeeze(),
        where="post",
        color="g",
        linestyle="--",
        linewidth=1,
        label="Salting(mg)",
    )
    ax3.set_ylabel("Salting/sanding (g/m²)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left")
    format_time_axis(ax3, dt_x, shared.av[0], day_tick_limit=150)

    plt.tight_layout()
    if shared.save_plots:
        plot_file_name = generate_plot_filename(
            title_str=paths.title_str,
            plot_type_flag=shared.av[0],
            figure_number=1,  # Traffic is figure 1
            plot_name="Traffic",
            date_num=shared.date_num,
            min_time=shared.i_min,
            max_time=shared.i_max,
        )
        plt.savefig(os.path.join(paths.path_outputfig, plot_file_name))
