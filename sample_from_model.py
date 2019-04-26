# Data
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

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
    actual_change_time = np.where(change_value == 1.0, np.nan, actual_change_time)

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
        # setting n=1 is equivalent to sampling from a Bernoulli distribution (no repeats)
        lick_samples = np.random.binomial(n=1, p=model_trial_output, size=(num_samples, len(model_trial_output)))
        model_rt = np.argmax(lick_samples, axis=1) # - actual_change_time[trial_num] # on each row, find the first occurence of 1 (lick)
        prop_lick = np.sum(model_rt > 0) / num_samples # proportion of lick, where reaction time is not zero.

        # Storing individual sample data
        individual_sample_rt.append(model_rt)
        individual_sample_actual_change_time.append(np.repeat(actual_change_time[trial_num], repeats=num_samples))
        individual_sample_change_value.append(np.repeat(change_value[trial_num], repeats=num_samples))

        # model_relative_rt = np.argmax(lick_samples, axis=1) - actual_change_time[trial_num]

        # mean reaction time (zero excluded)
        mean_model_rt = np.mean(model_rt[model_rt > 0])

        # compute model performance
        if not np.isnan(actual_change_time[trial_num]):
            early_lick_prop = np.sum(model_rt[model_rt > 0] < actual_change_time[trial_num]) / num_samples
            correct_lick_prop = np.sum(model_rt >= actual_change_time[trial_num]) / num_samples
            miss_prop = np.sum(model_rt == 0) / num_samples
            false_alarm_prop = np.nan
            correct_no_lick_prop = np.nan
        else:
            early_lick_prop = np.nan
            correct_lick_prop = np.nan
            miss_prop = np.nan
            false_alarm_prop = np.sum(model_rt > 0) / num_samples
            correct_no_lick_prop = np.sum(model_rt == 0) / num_samples

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
    sampled_data["mean_model_rt"] = mean_model_rt_list
    sampled_data["mean_model_relative_rt"] = mean_model_rt_list - actual_change_time

    sampled_data_df = pd.DataFrame.from_dict(sampled_data)

    individual_sample_dict = dict()
    # save data (flattening list of array with concat)
    individual_sample_dict["actual_change_time"] = np.concatenate(individual_sample_actual_change_time).ravel()
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

    # TODO: Think about subsetting criteria (eg. only hits)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_prop_choice["change_value"], df_prop_choice[metric], label="Model")
    ax.fill_between(df_prop_choice["change_value"], df_prop_choice[metric] - df_std[metric],
                    df_prop_choice[metric] + df_std[metric], alpha=0.3)
    ax.scatter(df_prop_choice["change_value"], df_prop_choice[metric], label=None, color="blue")

    # show all data points
    # ax.scatter(df["change_value"], df[metric], alpha=0.1, color="b")

    # Plot mouse behaviour as well
    if mouse_beahviour_df_path is not None:
        metric = "mouse_lick"
        mouse_df = pd.read_pickle(mouse_beahviour_df_path)
        mouse_prop_choice = mouse_df.groupby(["change"], as_index=False).agg({metric: "mean"})

        ax.plot(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[metric], label="Mouse")
        ax.scatter(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[metric], label=None)

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
    mouse_dict["mouse_lick"] = np.any([mouse_dict["mouse_hit"], mouse_dict["mouse_FA"]], axis=0).astype(float).flatten()


    mouse_df = pd.DataFrame.from_dict(mouse_dict)

    if savepath is not None:
        with open(savepath, "wb") as handle:
            pkl.dump(mouse_df, handle)
    else:
        return mouse_df


def compare_distributions(mouse_df_path, model_sample_path=None, savepath=None):
    """
    Raincloud plot to compare reaction time distributions
    :param mouse_df_path:
    :param model_sample_path:
    :param savepath:
    :return:
    """

    # plotting parameters
    strip_dot_size = 3
    strip_jitter = 0.5
    strip_dot_alpha = 1.0


    # TODO: write the same type of code for model (ideally generalised)

    if model_sample_path is None:
        mouse_behaviour_df = pd.read_pickle(mouse_df_path)
    else:
        model_behaviour_df = pd.read_pickle(model_sample_path)


    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    fig.subplots_adjust(wspace=0)

    # strip (ie. scatter with jitter)

    if model_sample_path is None:
        sns.stripplot(x="change", y="peri_stimulus_rt", data=mouse_behaviour_df.sample(1000),
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[0])
    else:
        sns.stripplot(x="change_value", y="mean_model_relative_rt", data=model_behaviour_df,
                      size=strip_dot_size, jitter=strip_jitter, alpha=strip_dot_alpha, ax=axs[0])

    # box plot
    # sns.boxplot(x="change", y="peri_stimulus_rt", data=mouse_behaviour_df, ax=axs[1])
    # distribution plot
    if model_sample_path is None:
        for change_val in np.unique(mouse_behaviour_df["change"]):
            index = np.where(mouse_behaviour_df["change"] == change_val)[0]
            data_to_plot = mouse_behaviour_df["peri_stimulus_rt"][index].dropna()
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


def main():

    model_number = 22
    mouse_number = 83

    # load data
    mainfolder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/")
    model_data_path = os.path.join(mainfolder, "hmm_data/model_response_083_22.pkl")
    model_sample_path = os.path.join(mainfolder, "hmm_data/model_samples_083_22.pkl")
    model_individual_sample_path = os.path.join(mainfolder, "hmm_data/model_individual_samples_083_" +
                                                str(model_number) + ".pkl")

    # Get mouse_df
    # mouse_behaviour_path = os.path.join(mainfolder, "exp_data/subsetted_data/data_IO_083.pkl")
    mouse_df_path = os.path.join(mainfolder, "mouse_83_df.pkl")
    # get_mouse_behaviour_df(mouse_behaviour_path, savepath=mouse_df_path)



    sample_from_model(model_data_path, num_samples=1000, plot_check=False,
                          savepath=model_sample_path, model_individual_sample_path=model_individual_sample_path)

    # plot_psychometric(model_sample_path, metric="prop_lick", label="Proportion of licks",
    #                  savepath=None, showfig=True)

    # Plot psychoemtric curve with model and mouse
    # figsavepath = os.path.join(mainfolder, "figures/psychometric_curve_comparision_mouse" + str(mouse_number)
    #                            + "_model_" + str(model_number))
    # plot_psychometric(model_sample_path, mouse_beahviour_df_path=mouse_df_path,
    #                   metric="prop_lick", label="Proportion of licks",
    #                    savepath=figsavepath, showfig=True)

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
    # figsavepath = os.path.join(mainfolder, "figures/reaction_time_distribution_mouse" + str(mouse_number))
    # compare_distributions(mouse_df_path, savepath=figsavepath)

    # figsavepath = os.path.join(mainfolder, "figures/reaction_time_distribution_model" + str(model_number))
    # compare_distributions(mouse_df_path=None, model_sample_path=model_sample_path, savepath=figsavepath)

    # Plot individual samples
    # figsavepath = os.path.join(mainfolder, "figures/individual_reaction_time_distribution_model" + str(model_number))
    # compare_distributions(mouse_df_path=model_individual_sample_path, savepath=figsavepath)


if __name__ == "__main__":
    main()