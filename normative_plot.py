# All the plotting functions
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Plotting single model evaluation

def plot_reaction_time_match(mouse_df, model_df, behaviour="FA", metric="absolute_decision_time"):
    # TODO: sort out where change_magnitude is
    if behaviour == "FA":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 & (mouse_df["change"] > 0))[0]
        mouse_df = mouse_df.loc[~mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 & (model_df["change"] > 0))[0]
        model_df = model_df.loc[~model_df.index.isin(model_hit_index)]
    elif behaviour == "Hit":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 &
                                   (np.exp(mouse_df["change"]) == change_magnitude))[0]
        mouse_df = mouse_df.loc[mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 &
                                   (np.exp(model_df["change"]) == change_magnitude))[0]
        model_df = model_df.loc[model_df.index.isin(model_hit_index)]

    model_df = model_df.groupby(["trial_ID"], as_index=False).agg({metric: "mean"})

    if behaviour == "FA":
        model_df = model_df.loc[~model_df.index.isin(mouse_hit_index)]# note the use of the mouse index

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mouse_df[metric], model_df[metric], alpha=0.3)
    ax.set_xlabel("Mouse reaction time")
    ax.set_ylabel("Model prediction")
    min_rt = min(np.concatenate([model_df[metric], mouse_df[metric]]))
    max_rt = max(np.concatenate([model_df[metric], mouse_df[metric]]))

    ax.set_xlim(min_rt, max_rt)
    ax.set_ylim(min_rt, max_rt)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    return fig


def compare_model_mouse_hazard_rate(model_hazard_rate, mouse_hazard_rate, scale_method=None):
    """
    Compare hazard rate in the experiment and hazard rate obtained from the model
    :param model_hazard_rate:
    :param mouse_hazard_rate:
    :param scale_method: if not None, then scales the model hazard rate
    :return:
    """
    # mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['font.family'] = 'Computer Modern Sans serif'

    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams['font.sans-serif'] = 'cm'

    # plt.style.use('gadfly')

    # params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    # matplotlib.rcParams.update(params)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(mouse_hazard_rate, label="Mouse")

    if scale_method == "softmax":
        model_hazard_rate = _softmax(model_hazard_rate)
        assert np.sum(model_hazard_rate) == 1.0, print("Softmax not summing to 1")
    elif scale_method == "max":
        model_hazard_rate = model_hazard_rate / max(model_hazard_rate)
    elif scale_method == "sum":
        model_hazard_rate = model_hazard_rate / np.sum(model_hazard_rate)

    ax.plot(model_hazard_rate, label="Model")
    ax.set_xlabel("Time from trial onset (frames)")
    ax.set_ylabel("")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.legend(frameon=False)



    return fig

# Plotting evaluation across models


def plot_multi_model_loss(model_comparison_df):
    # TODO: Make this into boxplot once multiple mice data are included
    fig, ax = plt.subplots(figsize=(8, 6))

    loss_metric = ["min_per_sample_train_loss", "min_per_sample_val_loss"] # "per_sample_test_loss"
    color_list = ["blue", "green"] # "blue"
    xoffset_list = [0.0, 0.25]

    for metric, col, xoffset in zip(loss_metric, color_list, xoffset_list):
        ax.bar(x=model_comparison_df["model_number"] + xoffset,
               height=model_comparison_df[metric],
               width=0.2, align="center", color=col)

    ax.set_ylabel("Loss per sample")
    ax.legend(["Training", "Validation"], frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig

def plot_multi_model_psychometric_error(model_comparison_df):

    fig, ax = plt.subplots(figsize=(8, 6))
    psychometric_metric = ["hit_exclude_FA_fit"]

    color_list = ["blue", "green"] # "blue"
    xoffset_list = [0.0, 0.25]

    for metric, col, xoffset in zip(psychometric_metric, color_list, xoffset_list):
        ax.bar(x=model_comparison_df["model_number"] + xoffset,
               height=model_comparison_df[metric],
               width=0.2, align="center", color=col)

    ax.set_ylabel("Mean-square error")
    ax.set_xlabel("Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_multi_model_rt_error(model_comparison_df):

    fig, ax = plt.subplots(figsize=(8, 6))

    rt_metric = ["FA_rt_dist_fit", "hit_rt_dist_fit_1", "hit_rt_dist_fit_1p25", "hit_rt_dist_fit_1p35",
                 "hit_rt_dist_fit_1p5", "hit_rt_dist_fit_2", "hit_rt_dist_fit_4"]
    xoffset_list = np.arange(-0.6, 0.8, 0.2)
    cmap = plt.get_cmap("Accent")
    cm_subsection = np.linspace(0, 1, len(rt_metric))
    color_list = [cmap(x) for x in cm_subsection]

    for metric, col, xoffset in zip(rt_metric, color_list, xoffset_list):
        ax.bar(x=model_comparison_df["model_number"] + xoffset,
               height=model_comparison_df[metric],
               width=0.2, align="center", color=col)


    ax.set_xlabel("Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Kolmogorov-Smirnov test statistic")
    ax.legend(["False alarm", "Hit: 1.0", "Hit: 1.25", "Hit: 1.35",
               "Hit: 1.5", "Hit: 2.0", "Hit: 4.0"], frameon=False)

# Helper functions


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)