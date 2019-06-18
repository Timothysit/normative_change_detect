import numpy as np
import matplotlib.pyplot as plt
import normative_plot as nmt
from hmm2_jax import create_vectorised_data
import seaborn as sns

import os
import pickle as pkl
import pandas as pd
from tqdm import tqdm

def find_pulses(exp_data, window_width=50, pulse_time_limit_seconds=None):
    """
    Find pulse locations from the signals.
    Fast pulse: 1.5 standard deviations above the mean temporal frequency (during baseline period)
    Slow pulse: 1.5 standard deviations below the mean temporal frequency (during baseline period)
    Intermediate pulse:
    :param signal_matrix:
    :return:
    """

    # TODO: Think about early vs. late fast vs. slow pulses (blocks)

    signals = exp_data["ys"]
    # change_times = exp_data["change"][0].flatten()

    signal_baseline = 0
    baseline_std = 0.25
    fast_slow_pulse_multiplier = 1.5
    fast_pulse_threshold = signal_baseline + (baseline_std * fast_slow_pulse_multiplier)
    slow_pulse_threshold = signal_baseline - (baseline_std * fast_slow_pulse_multiplier)

    # Reference pulse criteria
    reference_pulse_loc = list()
    reference_pulse_std_multiplier = 0.5
    reference_pulse_low_bound = signal_baseline - (baseline_std * reference_pulse_std_multiplier)
    reference_pulse_up_bound = signal_baseline + (baseline_std * reference_pulse_std_multiplier)

    # For loop implementation (there may be a vectorised way...)

    slow_pulse_loc = list()
    # intermediate_pulse_loc = list()
    fast_pulse_loc = list()

    num_trial = np.shape(exp_data["ys"])[0]
    for trial in np.arange(num_trial):
        trial_signal = signals[trial][0][0]
        change_time = exp_data["change"].flatten()[trial]
        trial_baseline_signal = trial_signal[:change_time-1]  # convert to 0-indexing!
        # trial_baseline_signal = trial_signal[:change_time - 1 - int(window_width/2)]  # remove pulse which overlap
        # with change time

        # further only subset pulse in the first 6 seconds
        if pulse_time_limit_seconds is not None:
            monitor_hz = 23
            pulse_time_limit_frames = monitor_hz * pulse_time_limit_seconds
            # further subset the baseline signal
            trial_baseline_signal = trial_baseline_signal[:pulse_time_limit_frames-1]  # 0-indexing again.

        fast_pulse_loc.append(np.where(trial_baseline_signal >= fast_pulse_threshold)[0])
        slow_pulse_loc.append(np.where(trial_baseline_signal <= slow_pulse_threshold)[0])
        reference_pulse_loc.append(np.where((trial_baseline_signal >= reference_pulse_low_bound) &
                                            (trial_baseline_signal <= reference_pulse_up_bound))[0])

    return slow_pulse_loc, fast_pulse_loc, reference_pulse_loc


def get_align_pulse_w_model_output(pulse_loc_list, model_output, window_width=10,
                               trial_subset=None, exclude_close_pulse=True):

    model_output_store = list()

    # peri_pulse_time = np.arange(0, window_width+1) - window_width/2

    # Naive implementation with loops (there is probably a better way to do this)
    if trial_subset is None:
        num_trial = len(pulse_loc_list)
    else:
        num_trial = trial_subset
    for trial in tqdm(np.arange(num_trial)):
        if exclude_close_pulse is True:
            if len(pulse_loc_list[trial]) >= 2:
                for n, pulse_interval in enumerate(np.diff(pulse_loc_list[trial])):
                    if pulse_interval <= window_width:
                        pulse_loc_list[trial] = np.delete(pulse_loc_list[trial], [n, n+1])
        for pulse_loc in pulse_loc_list[trial]:
            model_trial_output = model_output[trial, int(pulse_loc-window_width/2):int(pulse_loc+window_width/2)+1]
            if len(model_trial_output) == (window_width+1):
                model_output_store.append(model_trial_output)
                # ax.plot(peri_pulse_time, model_trial_output,
                #     alpha=0.1, color=linecolor)

    model_output_array = np.vstack(model_output_store)

    model_output_mean = np.nanmean(model_output_store, axis=0)
    model_output_std = np.nanstd(model_output_store, axis=0)

    return model_output_mean, model_output_std, model_output_array


def align_pulse_w_model_output(pulse_loc_list, model_output, window_width=10,
                               trial_subset=None, linecolor="blue", exclude_close_pulse=True,
                               intermediate_pulse_loc=None, fig=None, ax=None, linelabel=None,
                               shade_metric="std", linestyle="-"):
    """

    :param pulse_loc_list:
    :param model_output:
    :param window_width:
    :param trial_subset:
    :param linecolor:
    :param exclude_close_pulse: whether to exclude pulses that are within window_width of each other.
    :return:
    """
    if fig is None:
        fig, ax = plt.subplots()

    model_output_store = list()

    peri_pulse_time = np.arange(0, window_width+1) - window_width/2

    # Naive implementation with loops (there is probably a better way to do this)
    if trial_subset is None:
        num_trial = len(pulse_loc_list)
    else:
        num_trial = trial_subset
    for trial in tqdm(np.arange(num_trial)):
        if exclude_close_pulse is True:
            if len(pulse_loc_list[trial]) >= 2:
                for n, pulse_interval in enumerate(np.diff(pulse_loc_list[trial])):
                    if pulse_interval <= window_width:
                        pulse_loc_list[trial] = np.delete(pulse_loc_list[trial], [n, n+1])
        for pulse_loc in pulse_loc_list[trial]:
            model_trial_output = model_output[trial, int(pulse_loc-window_width/2):int(pulse_loc+window_width/2)+1]
            if len(model_trial_output) == (window_width+1):
                model_output_store.append(model_trial_output)
                # ax.plot(peri_pulse_time, model_trial_output,
                #     alpha=0.1, color=linecolor)

    # model_output_array = np.vstack(model_output_store)

    model_output_mean = np.nanmean(model_output_store, axis=0)
    model_output_std = np.nanstd(model_output_store, axis=0)

    if intermediate_pulse_loc is not None:
        intermediate_pulse_loc_mean, _, _ = get_align_pulse_w_model_output(intermediate_pulse_loc,
                                                                           model_output=model_output,
                                                                           window_width=window_width,
                                                                           trial_subset=None,
                                                                           exclude_close_pulse=exclude_close_pulse)
        model_output_mean = model_output_mean - intermediate_pulse_loc_mean

    ax.plot(peri_pulse_time, model_output_mean, color=linecolor, label=linelabel, linestyle=linestyle)

    if intermediate_pulse_loc is None:
        if shade_metric == "std":
            ax.fill_between(peri_pulse_time, model_output_mean - model_output_std,
                         model_output_mean + model_output_std, alpha=0.3, color=linecolor)


    ax.set_xlabel("Peri-pulse time (frames)")
    if intermediate_pulse_loc is not None:
        ax.set_ylabel(r"$P(z_\mathrm{change} \vert x_k) - P(z_\mathrm{change} \vert x_k)_\mathrm{reference}$")
    else:
        ax.set_ylabel(r"$P(z_\mathrm{change} \vert x_k)$")


    return fig, ax


def plot_model_pulse_response_dist(fig, ax, pulse_loc_list, model_output, color="blue"):
    num_trial = len(pulse_loc_list)
    model_pulse_response = list()

    for trial in tqdm(np.arange(num_trial)):
        for pulse_loc in pulse_loc_list[trial]:
            model_pulse_response.append(model_output[trial, pulse_loc])

    model_pulse_response = np.array(model_pulse_response)
    sns.kdeplot(model_pulse_response[~np.isnan(model_pulse_response)], ax=ax, color=color)

    return fig, ax

def main(model_number=70, mouse_number=75, time_shift=7, run_compare_early_late_block=False):
    home = os.path.expanduser("~")
    mouse_model_info = "_model_" + str(model_number) + "_mouse_" + str(mouse_number)
    additional_info = "before_change_window"
    print("Running alignment with mouse %d and model %d" % (mouse_number, model_number))
    # main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    main_folder = "/media/timothysit/180C-2DDD/second_rotation_project/"
    fig_folder = os.path.join(main_folder, "figures", "alignment_plots")

    # mouse_data_path = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(mouse_number) + ".pkl")
    mouse_data_path = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(mouse_number) +
                                   "_early_blocks" + ".pkl")
    with open(mouse_data_path, "rb") as handle:
        exp_data = pkl.load(handle)

    model_path = os.path.join(main_folder, "hmm_data/model_response_0" + str(mouse_number) + "_"
                                       + str(model_number) + "_time_shift_" + str(time_shift) + ".pkl")
    with open(model_path, "rb") as handle:
        model_data = pkl.load(handle)

    slow_pulse_loc, fast_pulse_loc, intermediate_pulse_loc = find_pulses(exp_data, window_width=50,
                                                                         pulse_time_limit_seconds=3)


    # plot pulse speed distribution just to double check
    figname = "pulse_speed_distribution" + mouse_model_info + "_first_1000"
    nmt.set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax = nmt.plot_pulse_speed_distribution(fig, ax, fast_pulse_loc=fast_pulse_loc[:1000],
                                                slow_pulse_loc=slow_pulse_loc[:1000],
                                                exp_data=exp_data, color=["blue", "orange"],
                                                plot_mean=True, plotrug=True)
    ax.set_xlim([-0.8, 0.8])
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)


    # plot pulse speed distribution of the intermediate pulse
    figname = "intermediate_pulse_speed_distribution_" + mouse_model_info + "_first_1000"
    nmt.set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax = nmt.plot_pulse_speed_distribution(fig, ax,
                                                intermediate_pulse_loc=intermediate_pulse_loc[:1000],
                                                exp_data=exp_data, color=["gray"],
                                                plot_mean=True, plotrug=True)
    ax.set_xlim([-0.8, 0.8])
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)


    # plot pulse time distribution
    figname = "pulse_time_distribution" + mouse_model_info + "_first_1000"
    fig, ax = plt.subplots(figsize=(4, 4))
    fig, ax = nmt.plot_pulse_time(fig, ax, fast_pulse_loc=fast_pulse_loc[:1000],
                                  slow_pulse_loc=slow_pulse_loc[:1000],
                              color=["blue", "orange"])
    ax.set_xlim([0, 350])
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    model_posterior_save_path = os.path.join(main_folder, "hmm_data", "model_posterior_mouse_"
                                             + str(mouse_number) + "_model_" + str(model_number) + ".pkl")
    with open(model_posterior_save_path, "rb") as handle:
        posterior_data = pkl.load(handle)


    # Plot posterior / model output to pulse
    fig, ax = plt.subplots()
    # model_output = model_data["model_vec_output"]
    model_output = posterior_data
    fig, ax = plot_model_pulse_response_dist(fig, ax, pulse_loc_list=fast_pulse_loc,
                                             model_output=model_output, color="blue")
    fig, ax = plot_model_pulse_response_dist(fig, ax, pulse_loc_list=slow_pulse_loc,
                                             model_output=model_output, color="orange")
    figname = "posterior_response_distribution" + mouse_model_info
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)


    # Plot posterior / model output aligned to pulse
    fig, ax = align_pulse_w_model_output(pulse_loc_list=fast_pulse_loc, model_output=model_output,
                                window_width=50, trial_subset=1000, linecolor="blue")
    ax.grid()
    fig.set_size_inches(4, 4)
    figname = "posterior_response_aligned_fast_pulse_mean_window50" + mouse_model_info
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    fig, ax = align_pulse_w_model_output(pulse_loc_list=slow_pulse_loc, model_output=model_output,
                                window_width=50, trial_subset=1000, linecolor="orange")
    ax.grid()
    figname = "posterior_response_aligned_slow_pulse_mean_window50" + mouse_model_info
    fig.set_size_inches(4, 4)
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)


    fig, ax = align_pulse_w_model_output(pulse_loc_list=intermediate_pulse_loc, model_output=model_output,
                                window_width=50, trial_subset=1000, linecolor="gray")
    ax.grid()
    figname = "posterior_response_aligned_intermediate_mean_window50" + mouse_model_info
    fig.set_size_inches(4, 4)
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    # Intermediate subtracted

    # Both fast and slow on the same plot

    fig, ax = plt.subplots()
    fig, ax = align_pulse_w_model_output(fig=fig, ax=ax, pulse_loc_list=fast_pulse_loc, model_output=model_output,
                                         intermediate_pulse_loc=intermediate_pulse_loc, exclude_close_pulse=False,
                                window_width=50, trial_subset=None, linecolor="blue", linelabel="Fast pulse")
    fig, ax = align_pulse_w_model_output(fig=fig, ax=ax, pulse_loc_list=slow_pulse_loc, model_output=model_output,
                                         intermediate_pulse_loc=intermediate_pulse_loc, exclude_close_pulse=False,
                                window_width=50, trial_subset=None, linecolor="orange", linelabel="Slow pulse")
    ax.grid()
    ax.legend()
    figname = "posterior_response_aligned_both_pulse1p5_intermediate0p5_subtracted_window50_include_close_pulse_all_" \
        + mouse_model_info
    fig.set_size_inches(4, 4)
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    # Intermediate subtracted pulse response separated by early vs. late blocks

    # compare early vs. late block pulse alignment

    if run_compare_early_late_block is True:
        print("Running comparison between early and late blocks")
        mouse_number = 75
        model_number_list = [74, 75]
        model_time_shift_list = [8, 9]
        line_styles = ["-", "--"]
        # exp_data_file_name_modifiers = ["_early_blocks", "_late_blocks"]
        exp_data_file_name_modifiers = ["_early_blocks", "_late_blocks"]
        # model_number_list = [74]
        # model_time_shift_list = [8]
        # line_styles = ["-"]
        # exp_data_file_name_modifiers = ["_early_blocks"]

        # figname = "posterior_response_aligned_early_vs_late_block_fast_pulse_intermediate_subtracted_corresponding_blocks_first_6_seconds"
        fig, ax = plt.subplots()
        for pulse_type in ["fast_pulse", "slow_pulse"]:
            figname = "posterior_response_aligned_early_vs_late_block_" + \
                      pulse_type + "_intermediate_subtracted_corresponding_blocks_first_6_seconds"
            for model_count, model_number in enumerate(model_number_list):
                print("Model number: " + str(model_number))
                print("Block type: " + exp_data_file_name_modifiers[model_count])
                exp_data_path = os.path.join(main_folder,
                                                           "exp_data/subsetted_data/data_IO_0" + str(mouse_number) +
                                                           exp_data_file_name_modifiers[model_count] + ".pkl")
                with open(exp_data_path, "rb") as handle:
                    exp_data = pkl.load(handle)

                slow_pulse_loc, fast_pulse_loc, intermediate_pulse_loc = find_pulses(exp_data, window_width=50,
                                                                                     pulse_time_limit_seconds=6)

                model_posterior_save_path = os.path.join(main_folder, "hmm_data", "model_posterior_mouse_"
                                                         + str(mouse_number) + "_model_" + str(model_number) + ".pkl")
                with open(model_posterior_save_path, "rb") as handle:
                    posterior_data = pkl.load(handle)

                if pulse_type == "fast_pulse":
                    pulse_loc = fast_pulse_loc
                elif pulse_type == "slow_pulse":
                    pulse_loc = slow_pulse_loc

                fig, ax = align_pulse_w_model_output(fig=fig, ax=ax, pulse_loc_list=pulse_loc,
                                                     intermediate_pulse_loc=intermediate_pulse_loc,
                                                     model_output=posterior_data,
                                                   window_width=50, trial_subset=1000, linecolor="blue",
                                                     shade_metric=None,
                                                     linestyle=line_styles[model_count])

        ax.grid()
        ax.legend(["early block", "late block"], frameon=False)
        fig.set_size_inches(4, 4)
        fig.savefig(os.path.join(fig_folder, figname), dpi=300)



if __name__ == "__main__":
    mouse_number_list = [75] # [75, 78, 79, 80, 81, 83]
    time_shift_list = [8] # [8, 9, 9, 10, 9, 8]
    for mouse_number, time_shift in zip(mouse_number_list, time_shift_list):
        main(model_number=74, mouse_number=mouse_number, time_shift=time_shift, run_compare_early_late_block=True)
