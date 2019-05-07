# Data
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# files
import os
import pickle as pkl
from os.path import expanduser
home = expanduser("~")

# util
from tqdm import tqdm


def sample_from_model(model_data_path, num_samples=1000, plot_check=False, savepath=None,
                      model_individual_sample_path=None):

    # samples discrete response from probabilistic output of model

    with open(model_data_path, "rb") as handle:
        model_data = pkl.load(handle)

    model_prob_output = model_data["model_vec_output"]
    actual_change_time = model_data["true_change_time"]
    change_value = model_data["change_value"]

    # when change goes from 1.0 to 1.0, there is no change and so change time should be nan
    # actual_change_time = np.where(change_value == 1.0, np.nan, actual_change_time)
    # ^ No longer valid if this is to be made consistent w/ mouse data

    # lists to store things
    prop_lick_list = list()
    early_lick_prop_list = list()
    correct_lick_prop_list = list()
    miss_prop_list = list()
    false_alarm_prop_list = list()
    correct_no_lick_list = list()
    mean_model_rt_list = list() # TODO: model reaction time relative to the stimulus change time

    # individual samples
    individual_sample_actual_change_time = list()
    individual_sample_rt = list()
    individual_sample_change_value = list()

    print("Sampling from model")
    for trial_num, model_trial_output in tqdm(enumerate(model_prob_output)):
        # don't read the nans from model_prob_output (in the case where model_prob_output is in matrix form)
        model_trial_output = model_trial_output[~np.isnan(model_trial_output)]

        # setting n=1 is equivalent to sampling from a Bernoulli distribution (no repeats)
        lick_samples = np.random.binomial(n=1, p=model_trial_output, size=(num_samples, len(model_trial_output)))
        model_no_lick_index = np.where(np.sum(lick_samples, axis=1) == 0)[0]
        model_rt = np.argmax(lick_samples, axis=1).astype(float) # - actual_change_time[trial_num] # on each row, find the first occurence of 1 (lick)
        # note that lick during the first frame will return model_rt of 0 (and so will no lick)
        # this problem is dealt with in the following line by setting true no licks to np.nan

        # also note that model reaction time is zero-indexed.
        # convert model reaction time to 1 indexed, so licking at frame 1 has model_rt = 1
        # IF USING TIME_SHIFTED SAMPLES, NEED TO GO BACK TO USING 0-INDEXING!
        model_rt = model_rt + 1

        model_rt[model_no_lick_index] = np.nan
        prop_lick = np.sum(~np.isnan(model_rt)) / num_samples # proportion of lick, where reaction time is not zero.

        # Storing individual sample data
        individual_sample_rt.append(model_rt)
        individual_sample_actual_change_time.append(np.repeat(actual_change_time[trial_num], repeats=num_samples))
        individual_sample_change_value.append(np.repeat(change_value[trial_num], repeats=num_samples))

        # model_relative_rt = np.argmax(lick_samples, axis=1) - actual_change_time[trial_num]

        # mean reaction time (zero excluded)
        mean_model_rt = np.nanmean(model_rt)


        # compute model performance
        """
        if not np.isnan(actual_change_time[trial_num]):
            early_lick_prop = np.sum(model_rt[~np.isnan(model_rt)] < actual_change_time[trial_num]) / num_samples
            correct_lick_prop = np.sum(model_rt >= actual_change_time[trial_num]) / num_samples
            miss_prop = np.sum(np.isnan(model_rt)) / num_samples
            false_alarm_prop = np.nan
            correct_no_lick_prop = np.nan
        else:
            early_lick_prop = np.nan
            correct_lick_prop = np.nan
            miss_prop = np.nan
            false_alarm_prop = np.sum(~np.isnan(model_rt)) / num_samples
            correct_no_lick_prop = np.sum(np.isnan(model_rt)) / num_samples
        """

        # under re-definition of hit and FA in trials with changes from baseline to baseline, there are no longer nan
        # actual change times, and early licks and FA becomes equivalent
        # and correct no licks become irrelevant (they are called "misses" in the no change trial)

        early_lick_prop = np.sum(model_rt[~np.isnan(model_rt)] < actual_change_time[trial_num]) / num_samples
        correct_lick_prop = np.sum(model_rt >= actual_change_time[trial_num]) / num_samples # equivalent to "Hit"
        miss_prop = np.sum(np.isnan(model_rt)) / num_samples
        false_alarm_prop = early_lick_prop
        correct_no_lick_prop = np.nan


        prop_lick_list.append(prop_lick)
        early_lick_prop_list.append(early_lick_prop)
        correct_lick_prop_list.append(correct_lick_prop)
        miss_prop_list.append(miss_prop)
        false_alarm_prop_list.append(false_alarm_prop)
        correct_no_lick_list.append(correct_no_lick_prop)
        mean_model_rt_list.append(mean_model_rt)


        # plot to check
        if plot_check is True:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(model_trial_output)
            sns.distplot(model_rt, ax=axs[1], kde=False)
            axs[1].axvline(mean_model_rt, color='b', linestyle='--')
            axs[1].axvline(actual_change_time[trial_num], color="r")

            axs[0].set_ylabel("P(lick)")
            axs[1].set_xlabel("Time (frames)")

            plt.show()

    # save summary stats of the sampled results for each trial

    sampled_data = dict()
    sampled_data["change_value"] = change_value
    sampled_data["actual_change_time"] = actual_change_time
    sampled_data["prop_lick"] = prop_lick_list
    sampled_data["early_lick_prop"] = early_lick_prop_list
    sampled_data["correct_lick_prop"] = correct_lick_prop_list
    sampled_data["miss_prop"] = miss_prop_list
    sampled_data["false_alarm_prop"] = false_alarm_prop_list
    sampled_data["correct_no_lick_prop"] = correct_no_lick_list
    # sampled_data["hit_prop"] = np.nansum([np.array(correct_lick_prop_list), np.array(correct_no_lick_list)], axis=0)
    # redefined "Hit" no longer counts no-licks during change from baseline to baseline
    sampled_data["hit_prop"] = correct_lick_prop_list
    sampled_data["mean_model_rt"] = mean_model_rt_list
    sampled_data["mean_model_relative_rt"] = mean_model_rt_list - actual_change_time

    sampled_data_df = pd.DataFrame.from_dict(sampled_data)

    individual_sample_dict = dict()
    # save data (flattening list of array with concat)
    individual_sample_dict["actual_change_time"] = np.concatenate(individual_sample_actual_change_time).ravel()
    individual_sample_dict["absolute_decision_time"] = np.concatenate(individual_sample_rt).ravel()
    individual_sample_dict["peri_stimulus_rt"] = np.concatenate(individual_sample_rt).ravel() - individual_sample_dict["actual_change_time"]
    individual_sample_dict["change"] = np.log(np.concatenate(individual_sample_change_value).ravel())
    individual_sample_df = pd.DataFrame.from_dict(individual_sample_dict)

    if savepath is not None:
        with open(savepath, "wb") as handle:
            pkl.dump(sampled_data_df, handle)

    if model_individual_sample_path is not None:
        with open(model_individual_sample_path, "wb") as handle:
            pkl.dump(individual_sample_df, handle)

def plot_psychometric(model_sample_path, mouse_beahviour_df_path=None, metric="prop_lick", label="Proportion of licks",
                      savepath=None, showfig=True):
    df = pd.read_pickle(model_sample_path)
    df_prop_choice = df.groupby(["change_value"], as_index=False).agg({metric: "mean"})
    df_std = df.groupby(["change_value"], as_index=False).agg({metric: "std"})

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_prop_choice["change_value"], df_prop_choice[metric], label="Model")
    ax.fill_between(df_prop_choice["change_value"], df_prop_choice[metric] - df_std[metric],
                    df_prop_choice[metric] + df_std[metric], alpha=0.3)
    ax.scatter(df_prop_choice["change_value"], df_prop_choice[metric], label=None, color="blue")

    # show all data points
    # ax.scatter(df["change_value"], df[metric], alpha=0.1, color="b")

    # Plot mouse behaviour as well
    if mouse_beahviour_df_path is not None:
        if metric == "prop_lick":
            mouse_metric = "mouse_lick"
        elif metric == "hit_prop":
            mouse_metric = "mouse_hit"
        elif metric == "false_alarm_prop":
            mouse_metric = "mouse_FA"
        elif metric == "miss_prop":
            mouse_metric = "mouse_miss"
        mouse_df = pd.read_pickle(mouse_beahviour_df_path)
        mouse_prop_choice = mouse_df.groupby(["change"], as_index=False).agg({mouse_metric: "mean"})

        ax.plot(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[mouse_metric], label="Mouse")
        ax.scatter(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[mouse_metric], label=None)

        ax.legend(frameon=False)

    if "prop" in metric:
        ax.set_ylim([0, 1.05])

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel(label)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()



def compare_model_w_behaviour_plot(model_sample_path, mouse_df_path,
                                   metric="prop_lick", label="Proportion of licks", figsavepath=None):

    with open(mouse_df_path, "rb") as handle:
        mouse_df = pkl.load(handle)


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    mouse_prop_choice = mouse_df.groupby(["change"], as_index=False).agg({metric: "mean"})

    ax.plot(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[metric], label="mouse")
    ax.scatter(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[metric], label=None)

    if "rt" in metric:
        ax.scatter(np.exp(mouse_df["change"]), mouse_df[metric], label=None, alpha=0.5)


    ax.set_xlabel("Change")
    ax.set_ylabel(label)

    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath)

    plt.show()







def get_mouse_behaviour_df(mouse_behaviour_path, savepath=None):
    """
    convert data from dictionary to dataframe format, and a bit of data reorganiation.
    :param mouse_behaviour_path:
    :return:
    """

    with open(mouse_behaviour_path, "rb") as handle:
        exp_data = pkl.load(handle)

    mouse_dict = dict()

    mouse_dict["change"] = exp_data["sig"].flatten() # change magnitude
    mouse_dict["actual_change_time"] = exp_data["change"].flatten()
    mouse_dict["absolute_decision_time"] = exp_data["rt"].flatten()
    mouse_dict["peri_stimulus_rt"] = mouse_dict["absolute_decision_time"] - mouse_dict["actual_change_time"]

    # Experimental data
    mouse_dict["mouse_hit"] = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_dict["mouse_FA"] = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_dict["mouse_miss"] = (exp_data["outcome"] == "Miss").astype(float).flatten()
    # below, I use multiplication of two binary vectors in place of the "&" operation
    mouse_dict["correct_lick"] = (mouse_dict["change"] > 0).astype(float) * mouse_dict["mouse_hit"]

    # TODO: Also look at correct_no_lick

    # "Hit" during no change period is also a lick (as explained by Petr 2019-04-29, see notes)
    # mouse_dict["mouse_lick"] = mouse_dict["correct_lick"] + mouse_dict["mouse_FA"] # AND operation
    mouse_dict["mouse_lick"] = mouse_dict["mouse_hit"] + mouse_dict["mouse_FA"]

    # mouse_dict["mouse_lick"] = np.any([mouse_dict["mouse_hit"], mouse_dict["mouse_FA"]], axis=0).astype(float).flatten() # this line is wrong, some hits can be no-licks.

    mouse_df = pd.DataFrame.from_dict(mouse_dict)

    if savepath is not None:
        with open(savepath, "wb") as handle:
            pkl.dump(mouse_df, handle)
    else:
        return mouse_df


def compare_distributions(mouse_df_path, model_sample_path=None, savepath=None, sub_sample=None, plot_type="hits",
                          metric="peri_stimulus_rt", ylabel="Peri stimulus reaction time (frames)"):
    """
    Raincloud plot to compare reaction time distributions
    :param mouse_df_path:
    :param model_sample_path:
    :param savepath:
    :param metric: type of reaction time to plot,
        option 1 | "peri_stimulus_rt"
        option 2 | "absolute_decision_time"
    :return:
    """

    # plotting parameters
    strip_dot_size = 3
    strip_jitter = 0.5
    strip_dot_alpha = 1.0


    # TODO: write the same type of code for model (ideally generalised)

    if model_sample_path is None:
        mouse_behaviour_df = pd.read_pickle(mouse_df_path)
        if plot_type == "hits":
            subset_index = np.where((mouse_behaviour_df["peri_stimulus_rt"] >= 0) & (mouse_behaviour_df["change"] > 0))
            mouse_behaviour_df = mouse_behaviour_df.iloc[subset_index]
        elif plot_type == "false_alarms":
            hit_index = np.where((mouse_behaviour_df["peri_stimulus_rt"]) >= 0 & (mouse_behaviour_df["change"] > 0))[0]
            mouse_behaviour_df = mouse_behaviour_df.loc[~mouse_behaviour_df.index.isin(hit_index)]
            # nice trick from: https://stackoverflow.com/questions/29134635/slice-pandas-dataframe-by-labels-that-are-not-in-a-list
    else:
        model_behaviour_df = pd.read_pickle(model_sample_path)





    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    fig.subplots_adjust(wspace=0)

    # strip (ie. scatter with jitter)

    if model_sample_path is None:
        if sub_sample is None:
            sns.stripplot(x="change", y=metric, data=mouse_behaviour_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[0])
        else:
            sns.stripplot(x="change", y=metric, data=mouse_behaviour_df.sample(sub_sample),
                          size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[0])
    else:
        sns.stripplot(x="change_value", y=metric, data=model_behaviour_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[0])

    # box plot
    # sns.boxplot(x="change", y="peri_stimulus_rt", data=mouse_behaviour_df, ax=axs[1])
    # distribution plot
    if model_sample_path is None:
        for change_val in np.unique(mouse_behaviour_df["change"]):
            index = np.where(mouse_behaviour_df["change"] == change_val)[0]
            data_to_plot = mouse_behaviour_df[metric][index] .dropna()
            if len(data_to_plot) == 0:
                pass
            else:
                sns.kdeplot(data_to_plot, shade=True, linewidth=0, vertical=True, alpha=0.5, ax=axs[1])
    else:
        for change_val in np.unique(model_behaviour_df["change_value"]):
            index = np.where(model_behaviour_df["change_value"] == change_val)[0]
            # sns.kdeplot(model_behaviour_df["mean_model_relative_rt"][index], shade=True, linewidth=0, vertical=True, alpha=0.5, ax=axs[1])
            data_to_plot = model_behaviour_df["mean_model_relative_rt"][index].dropna()
            if len(data_to_plot) == 0:
                pass
            else:
                sns.kdeplot(model_behaviour_df["mean_model_relative_rt"][index], shade=True, linewidth=0, vertical=True,
                            alpha=0.5, ax=axs[1])


    if model_sample_path is None:
        axs[0].legend(np.exp(np.unique(mouse_behaviour_df["change"])), frameon=False, title="Change magnitude",
                      markerscale=2.0, bbox_to_anchor=(2.0, 0.5))
    else:
        axs[0].legend(np.unique(model_behaviour_df["change_value"]), frameon=False, title="Change magnitude",
                      markerscale=2.0, bbox_to_anchor=(2.0, 0.5))

    axs[1].get_legend().remove()
    axs[0].set_zorder(1) # make axis 0 go to front (and so legend on top of axis 1)

    axs[0].set_ylabel("Peri-stimulus reaction time (frames)")

    for ax_placeholder in axs:
        ax_placeholder.get_xaxis().set_visible(False)
        ax_placeholder.spines["bottom"].set_visible(False)
        ax_placeholder.spines["right"].set_visible(False)
        ax_placeholder.spines["left"].set_visible(False)
        ax_placeholder.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    plt.show()


def compare_model_and_mouse_dist(mouse_df_path, model_df_path, plot_type="FA", change_magnitude=None, savepath=None,
                                 strip_dot_size=2, strip_jitter=1, strip_dot_alpha=1, model_sub_sample=1000,
                                 show_model_sub_sample=None, showfig=True):

    mouse_df = pd.read_pickle(mouse_df_path)
    model_df = pd.read_pickle(model_df_path)


    if plot_type == "FA":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 & (mouse_df["change"] > 0))[0]
        mouse_df = mouse_df.loc[~mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 & (model_df["change"] > 0))[0]
        model_df = model_df.loc[~model_df.index.isin(model_hit_index)]
    elif plot_type == "Hit":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 & (np.exp(mouse_df["change"]) == change_magnitude))[0]
        mouse_df = mouse_df.loc[mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 & (np.exp(model_df["change"]) == change_magnitude))[0]
        model_df = model_df.loc[model_df.index.isin(model_hit_index)]


    if model_sub_sample is not None:
        model_df = model_df.sample(model_sub_sample)

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    axs = [ax1, ax2, ax3]

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)



    # fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)



    if plot_type == "FA":
        # first row: KDE of model and mouse
        sns.kdeplot(mouse_df["absolute_decision_time"], shade=False, linewidth=3, vertical=False, alpha=0.5, ax=axs[0],
                    label="Mouse")
        sns.kdeplot(model_df["absolute_decision_time"], shade=False, linewidth=3, vertical=False, alpha=0.5, ax=axs[0],
                    label="Model")
        axs[0].legend(frameon=False)

        # second row: mouse reaction times
        sns.stripplot(x="absolute_decision_time", data=mouse_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[1],
                      color="blue")


        # third row: model reaction times
        if show_model_sub_sample is not None:
            model_df = model_df.sample(show_model_sub_sample)

        sns.stripplot(x="absolute_decision_time", data=model_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[2],
                      color="orange")


        for ax in axs:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        axs[0].spines["bottom"].set_visible(False)
        axs[1].spines["bottom"].set_visible(False)

        axs[1].spines["left"].set_visible(False)
        axs[2].spines["left"].set_visible(False)

        axs[2].set_xlabel("Time to lick (frames)")
        axs[1].set_xlabel("")

        ax1.set_xlim(xmin=0)
    # TODO: There is significant overlap in the two conditions, can remove duplicates
    elif plot_type == "Hit":
        # first row: KDE of model and mouse
        sns.kdeplot(mouse_df["peri_stimulus_rt"], shade=False, linewidth=3, vertical=False, alpha=0.5, ax=axs[0],
                    label="Mouse")
        sns.kdeplot(model_df["peri_stimulus_rt"], shade=False, linewidth=3, vertical=False, alpha=0.5, ax=axs[0],
                    label="Model")
        axs[0].legend(frameon=False)

        # second row: mouse reaction times
        sns.stripplot(x="peri_stimulus_rt", data=mouse_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[1],
                      color="blue")


        # third row: model reaction times
        if show_model_sub_sample is not None:
            model_df = model_df.sample(show_model_sub_sample)

        sns.stripplot(x="peri_stimulus_rt", data=model_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[2],
                      color="orange")


        for ax in axs:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        axs[0].spines["bottom"].set_visible(False)
        axs[1].spines["bottom"].set_visible(False)

        axs[1].spines["left"].set_visible(False)
        axs[2].spines["left"].set_visible(False)

        axs[2].set_xlabel("Peri-stimulus lick time (frame)")
        axs[1].set_xlabel("")

        ax1.set_xlim(xmin=0)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()


def main(model_number=20, mouse_number=83, run_get_mouse_df=True, run_sample_from_model=True, run_plot_psychom_metric_comparison=True,
         run_plot_mouse_reaction_time=True,
         run_plot_model_reaction_time=True, run_compare_model_and_mouse_dist=False):

    # load data
    mainfolder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/")
    model_data_path = os.path.join(mainfolder, "hmm_data/model_response_0" + str(mouse_number) +
                                   "_" + str(model_number) + str(".pkl"))
    model_sample_path = os.path.join(mainfolder, "hmm_data/model_samples_0" + str(mouse_number) +
                                    "_" + str(model_number) + str(".pkl"))
    model_individual_sample_path = os.path.join(mainfolder, "hmm_data/model_individual_samples_083_" +
                                                str(model_number) + ".pkl")

    # Get mouse_df

    mouse_behaviour_path = os.path.join(mainfolder, "exp_data/subsetted_data/data_IO_083.pkl")
    mouse_df_path = os.path.join(mainfolder, "mouse_83_df.pkl")

    if run_get_mouse_df is True:
        get_mouse_behaviour_df(mouse_behaviour_path, savepath=mouse_df_path)


    if run_sample_from_model is True:
        sample_from_model(model_data_path, num_samples=1000, plot_check=False,
                         savepath=model_sample_path, model_individual_sample_path=model_individual_sample_path)

    # plot_psychometric(model_sample_path, metric="prop_lick", label="Proportion of licks",
    #                  savepath=None, showfig=True)

    # Plot psychometric curve with model and mouse
    if run_plot_psychom_metric_comparison is True:
        figsavepath = os.path.join(mainfolder, "figures/licks_psychometric_curve_comparision_mouse" + str(mouse_number)
                                    + "_model_" + str(model_number))
        plot_psychometric(model_sample_path, mouse_beahviour_df_path=mouse_df_path,
                           metric="prop_lick", label="Proportion of licks",
                            savepath=figsavepath, showfig=True)

        figsavepath = os.path.join(mainfolder, "figures/hits_psychometric_curve_comparision_mouse" + str(mouse_number)
                                    + "_model_" + str(model_number))
        plot_psychometric(model_sample_path, mouse_df_path,
                                      metric="hit_prop", label="Proportion of hits", savepath=figsavepath)

        # False Alarms
        figsavepath = os.path.join(mainfolder, "figures/FA_psychometric_curve_comparision_mouse" + str(mouse_number)
                                    + "_model_" + str(model_number))
        plot_psychometric(model_sample_path, mouse_df_path,
                      metric="false_alarm_prop", label="Proportion of false alarms", savepath=figsavepath)

        # Misses
        figsavepath = os.path.join(mainfolder, "figures/Miss_psychometric_curve_comparision_mouse" + str(mouse_number)
                                    + "_model_" + str(model_number))
        plot_psychometric(model_sample_path, mouse_df_path,
                      metric="miss_prop", label="Proportion of misses", savepath=figsavepath)



    # savepath = os.path.join(mainfolder, "reaction_time_samples_model_" + str(model_number))
    # plot_psychometric(model_sample_path, metric="
    #
    # ", label="Mean reaction time",
    #                    savepath=savepath, showfig=True)

    # Compare samples from model w/ mice behaviour: Proportion licks
    # compare_model_w_behaviour_plot(model_sample_path, mouse_df_path,
    #                                metric="mouse_lick", label="Proportion of licks", figsavepath=None)

    # compare_model_w_behaviour_plot(model_sample_path, mouse_df_path,
    #                               metric="mouse_hit", label="Proportion of hits", figsavepath=None)

    # compare_model_w_behaviour_plot(model_sample_path, mouse_df_path,
    #                                metric="mouse_FA", label="Proportion of false alarms", figsavepath=None)

    # Compare samples from model w/ mice behaviour: Reaction time
    # figsavepath = os.path.join(mainfolder, "reaction_time_mouse_" + str(mouse_number))
    # compare_model_w_behaviour_plot(model_sample_path, mouse_df_path,
    #                                metric="peri_stimulus_rt", label="Reaction time (frames)", figsavepath=None)


    # Plot reaction time distributions
    if run_plot_mouse_reaction_time is True:
        figsavepath = os.path.join(mainfolder, "figures/reaction_time_distribution_mouse_hits_" + str(mouse_number))
        compare_distributions(mouse_df_path, savepath=figsavepath, plot_type="hits")

        figsavepath = os.path.join(mainfolder, "figures/reaction_time_distribution_false_alarms_" + str(mouse_number))
        compare_distributions(mouse_df_path, savepath=figsavepath, plot_type="false_alarms", metric="absolute_decision_time")

    #figsavepath = os.path.join(mainfolder, "figures/reaction_time_distribution_model" + str(model_number))
    # compare_distributions(mouse_df_path=None, model_sample_path=model_sample_path, savepath=figsavepath)

    # Plot individual samples of reaction time from the model
    if run_plot_model_reaction_time is True:
        figsavepath = os.path.join(mainfolder, "figures/individual_reaction_time_distribution_model_hits_"
                                   + str(model_number))
        compare_distributions(mouse_df_path=model_individual_sample_path, savepath=figsavepath, sub_sample=1000,
                              plot_type="hits")

        figsavepath = os.path.join(mainfolder, "figures/individual_reaction_time_distribution_model_false_alarms_"
                                   + str(model_number))
        compare_distributions(mouse_df_path=model_individual_sample_path, savepath=figsavepath, sub_sample=1000,
                              plot_type="false_alarms", metric="absolute_decision_time")

    if run_compare_model_and_mouse_dist is True:
        additional_info = "sub_sample_10000"
        figsavepath = os.path.join(mainfolder, "figures/reaction_time_false_alarm_dist_comparison_model_"
                                   + str(model_number) + additional_info + ".png")
        compare_model_and_mouse_dist(mouse_df_path=mouse_df_path, model_df_path=model_individual_sample_path, 
                                     savepath=figsavepath, plot_type="FA", strip_dot_size=2, strip_jitter=1.2,
                                     strip_dot_alpha=0.6, model_sub_sample=10000,
                                     show_model_sub_sample=None)

        for change_magnitude in [1.25, 1.35, 1.50, 2.00, 4.00]:
            figsavepath = os.path.join(mainfolder, "figures/reaction_time_hits_dist_comparison_model_"
                                       + str(model_number) + "change_" + str(change_magnitude) + additional_info + ".png")
            compare_model_and_mouse_dist(mouse_df_path=mouse_df_path, model_df_path=model_individual_sample_path,
                                         savepath=figsavepath, plot_type="Hit", strip_dot_size=2, strip_jitter=1.2,
                                         strip_dot_alpha=0.6, model_sub_sample=10000,
                                         change_magnitude=change_magnitude,
                                         show_model_sub_sample=None, showfig=False)



if __name__ == "__main__":
    main(model_number=999, run_get_mouse_df=False, run_sample_from_model=True, run_plot_psychom_metric_comparison=False,
         run_plot_mouse_reaction_time=False,
         run_plot_model_reaction_time=False,
         run_compare_model_and_mouse_dist=False)