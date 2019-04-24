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


def sample_from_model(model_data_path, num_samples=1000, plot_check=False, savepath=None):

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
    mean_model_rt_list = list()

    print("Sampling from model")
    for trial_num, model_trial_output in tqdm(enumerate(model_prob_output)):
        # setting n=1 is equivalent to sampling from a Bernoulli distribution (no repeats)
        lick_samples = np.random.binomial(n=1, p=model_trial_output, size=(num_samples, len(model_trial_output)))
        model_rt = np.argmax(lick_samples, axis=1) # on each row, find the first occurence of 1 (lick)
        prop_lick = np.sum(model_rt > 0) / num_samples # proportion of lick, where reaction time is not zero.

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

    sampled_data_df = pd.DataFrame.from_dict(sampled_data)

    if savepath is not None:
        with open(savepath, "wb") as handle:
            pkl.dump(sampled_data_df, handle)

def plot_psychometric(model_sample_path, metric="prop_lick", label="Proportion of licks", savepath=None, showfig=True):
    df = pd.read_pickle(model_sample_path)
    df_prop_choice = df.groupby(["change_value"], as_index=False).agg({metric: "mean"})
    df_std = df.groupby(["change_value"], as_index=False).agg({metric: "std"})

    # TODO: Also get the standard deviation

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_prop_choice["change_value"], df_prop_choice[metric])
    ax.fill_between(df_prop_choice["change_value"], df_prop_choice[metric] - df_std[metric],
                    df_prop_choice[metric] + df_std[metric], alpha=0.3)
    ax.scatter(df_prop_choice["change_value"], df_prop_choice[metric])

    # show all data points
    ax.scatter(df["change_value"], df[metric], alpha=0.1, color="b")

    ax.set_ylim([0, 1.05])

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel(label)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()






def main():

    # load data
    mainfolder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data")
    model_data_path = os.path.join(mainfolder, "model_response_083_20.pkl")
    model_sample_path = os.path.join(mainfolder, "model_samples_083_20.pkl")

    sample_from_model(model_data_path, num_samples=1000, plot_check=True,
                       savepath=model_sample_path)

    plot_psychometric(model_sample_path, metric="prop_lick", label="Proportion of licks",
                      savepath=None, showfig=True)



if __name__ == "__main__":
    main()