import numpy as np
import matplotlib.pyplot as plt
import normative_plot as nmt
from hmm2_jax import create_vectorised_data
import seaborn as sns

import os
import pickle as pkl
import pandas as pd
from tqdm import tqdm

def find_pulses(exp_data, window_width=50):
    """
    Find pulse locations from the signals.
    Fast pulse: 1.5 standard deviations above the mean temporal frequency (during baseline period)
    Slow pulse: 1.5 standard deviations below the mean temporal frequency (during baseline period)

    :param signal_matrix:
    :return:
    """

    # TODO: Think about early vs. late fast vs. slow pulses (blocks)

    signals = exp_data["ys"]
    # change_times = exp_data["change"][0].flatten()

    signal_baseline = 0
    baseline_std = 0.25
    fast_pulse_threshold = signal_baseline + (baseline_std * 0.5)
    slow_pulse_threshold = signal_baseline - (baseline_std * 0.5)

    # Reference pulse criteria
    reference_pulse_loc = list()
    reference_pulse_std_multiplier = 1.5
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
        # trial_baseline_signal = trial_signal[:change_time - 1 - int(window_width/2)]
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
                               intermediate_pulse_loc=None, fig=None, ax=None, linelabel=None):
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

    ax.plot(peri_pulse_time, model_output_mean, color=linecolor, label=linelabel)

    if intermediate_pulse_loc is None:
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

def main(model_number=70, mouse_number=75, time_shift=7):
    home = os.path.expanduser("~")
    mouse_model_info = "_model_" + str(model_number) + "_mouse_" + str(mouse_number)
    additional_info = "before_change_window"
    print("Running alignment with mouse %d and model %d" % (mouse_number, model_number))
    main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    fig_folder = os.path.join(main_folder, "figures", "alignment_plots")

    mouse_data_path = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(mouse_number) + ".pkl")
    with open(mouse_data_path, "rb") as handle:
        exp_data = pkl.load(handle)

    model_path = os.path.join(main_folder, "hmm_data/model_response_0" + str(mouse_number) + "_"
                                       + str(model_number) + "_time_shift_" + str(time_shift) + ".pkl")
    with open(model_path, "rb") as handle:
        model_data = pkl.load(handle)

    slow_pulse_loc, fast_pulse_loc, intermediate_pulse_loc = find_pulses(exp_data, window_width=50)

    # plot pulse speed distribution just to double check
    figname = "pulse_speed_distribution" + mouse_model_info
    nmt.set_style()
    fig, ax = plt.subplots(figsize=(4, 4))
    fig, ax = nmt.plot_pulse_speed_distribution(fig, ax, fast_pulse_loc=fast_pulse_loc, slow_pulse_loc=slow_pulse_loc,
                                                exp_data=exp_data, color=["blue", "orange"])
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    # plot pulse time distribution
    figname = "pulse_time_distribution" + mouse_model_info
    fig, ax = plt.subplots(figsize=(4, 4))
    fig, ax = nmt.plot_pulse_time(fig, ax, fast_pulse_loc=fast_pulse_loc, slow_pulse_loc=slow_pulse_loc,
                              color=["blue", "orange"])
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
    figname = "posterior_response_aligned_both_pulse_intermediate_subtracted_window50_include_close_pulse" \
        + mouse_model_info
    fig.set_size_inches(4, 4)
    fig.savefig(os.path.join(fig_folder, figname), dpi=300)

    # Intermediate subtracted pulse response separated by early vs. late blocks


if __name__ == "__main__":
    mouse_number_list = [75, 78, 79, 80, 81, 83]
    time_shift_list = [6, 7, 9, 8, 7, 7]
    for mouse_number, time_shift in zip(mouse_number_list, time_shift_list):
        main(model_number=67, mouse_number=mouse_number, time_shift=time_shift)