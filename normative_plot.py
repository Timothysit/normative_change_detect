# All the plotting functions
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import random as baserandom # for random choice of signals/posterior to plot
import pickle as pkl
import pandas as pd
import seaborn as sns

# Plotting single model evaluation

def plot_training_loss(training_savepath, figsavepath=None, cv=False, time_shift=False):
    """
    Plots training and validation loss across epochs for a particular model.
    :param training_savepath:
    :param figsavepath:
    :param cv:
    :param time_shift:
    :return:
    """

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    epoch_per_step = 10
    if cv is False:
        loss = training_result["loss"]
    else:
        train_loss = training_result["train_loss"]
        val_loss = training_result["val_loss"]
        epoch = training_result["epoch"]
        train_set_size = training_result["train_set_size"]
        val_set_size = training_result["val_set_size"]

        train_loss = np.hstack(train_loss)
        val_loss = np.hstack(val_loss)

    parameters = training_result["param_val"]

    if cv is False:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(np.arange(1, len(loss)+1) * epoch_per_step, loss)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    elif cv is True and time_shift is False:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(epoch, train_loss / train_set_size, label="Training loss")
        ax1.set_xlabel("Epoch")
        # ax1.set_ylabel("Training loss", color="tab:blue")
        # ax2 = ax1.twinx()
        # ax2.plot(epoch, val_loss / val_set_size, color="red")
        # ax2.set_ylabel("Validation loss", color="tab:red")

        ax1.plot(epoch, val_loss / val_set_size, label="Validation loss")
        ax1.set_ylabel("Mean loss")
        ax1.legend(frameon=False)
        fig.tight_layout()
    elif cv is True and time_shift is True:
        fig, ax = plt.subplots(2, 5, sharey=True, sharex=True)
        ax = ax.flatten()
        time_shift_list = training_result["time_shift"]
        for n, time_shift in enumerate(onp.unique(time_shift_list)):
            time_shift_index = onp.where(time_shift_list == time_shift)[0]
            ax[n].plot(train_loss[time_shift_index] / train_set_size, label="Training loss")
            ax[n].plot(val_loss[time_shift_index] / val_set_size, label="Validation loss")
            ax[n].set_title("Time shift:" + str(time_shift))
            ax[n].set_ylabel("Mean loss")

    if cv is False:
        print("Minimum training loss:", str(min(loss)))
        print("Epoch:", str(np.argmin(np.array(loss))))
    else:
        print("Minimum validation loss:", str(min(val_loss)))
        print("Minimum mean validation loss:", str(min(val_loss) / val_set_size))
        print("Epoch:", str(np.argmin(np.array(val_loss))))


    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()

def plot_sigmoid_comparisions(training_savepath, plot_least_loss_sigmoid=False, figsavepath=None):
    """
    Plot tunned sigmoid over epochs.
    :param training_savepath:
    :param plot_least_loss_sigmoid:
    :param figsavepath:
    :return:
    """

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    parameters = training_result["param_val"]
    num_set_to_plot = 10
    epoch_step_size = 10
    total_num_set = len(parameters)
    set_index_to_plot = onp.round(onp.linspace(0, total_num_set-1, num_set_to_plot))
    epochs_plotted = (set_index_to_plot + 1) * epoch_step_size

    # only plot the psychometric with the least cost
    if plot_least_loss_sigmoid is True:
        set_index_to_plot = onp.where(training_result["loss"] == min(training_result["loss"]))[0]
        epochs_plotted = set_index_to_plot

    fake_posterior = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for index in set_index_to_plot:

        param_vals = parameters[int(index)]

        p_lick = apply_cost_benefit(change_posterior=fake_posterior, true_positive=1.0, true_negative=0.0,
                                    false_negative=0.0, false_positive=0.0) # param_vals[1]
        p_lick = apply_strategy(p_lick, k=param_vals[0], midpoint=param_vals[1])

        ax.plot(fake_posterior, p_lick)

    ax.legend(list(epochs_plotted), frameon=False, title="Epochs")
    ax.set_xlabel(r"$p(z_t = \mathrm{change} \vert x_{1:t})$")
    ax.set_ylabel(r"$p(\mathrm{lick})$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


def plot_trained_hazard_rate(training_savepath, sigmoid_function=None, num_non_hazard_rate_param=2):
    """
    Plot fitted hazard rate.
    :param training_savepath:
    :param sigmoid_function: sigmoid function used to convert between parameter and actual hazard rate
    :param num_non_hazard_rate_param:
    :return:
    """
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    epoch_index = np.where(training_result["val_loss"] == np.nanmin(training_result["val_loss"]))[0][0]
    epoch_param = training_result["param_val"][epoch_index]
    hazard_rate = epoch_param[num_non_hazard_rate_param:]
    hazard_rate = hazard_rate[:-10]  # remove the last few values, which are usually not runned through gradient descent

    plt.style.use("~/Dropbox/notes/Projects/second_rotation_project/normative_model/ts.mplstyle")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if sigmoid_function is None:
        ax.plot(hazard_rate)
    else:
        ax.plot(sigmoid_function(hazard_rate))
    # ax.plot(softmax(hazard_rate))
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("P(change)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.grid()

    return fig, ax

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

def plot_time_shift_test(exp_data_path, param=[10, 0.5], time_shift_list=[0, 1], trial_num=0):

    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path)

    # subset to smaller size to compute faster
    signal_matrix = signal_matrix[0:2, :]
    lick_matrix = lick_matrix[0:2, :]

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.plot(lick_matrix[trial_num, :])

    global time_shift
    for time_shift in time_shift_list:
        batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))
        prediction_matrix = batched_predict_lick(param, signal_matrix)
        batch_loss = matrix_cross_entropy_loss(lick_matrix, prediction_matrix)

        axs.plot(prediction_matrix[trial_num, :])
        print("Loss: ", str(batch_loss))

    plt.show()


def plot_trained_posterior(datapath, training_savepath, num_examples=10, random_seed=777,
                           plot_peri_stimulus=True, figsavepath=None, plot_cumulative=False):

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    change_values = np.exp(exp_data["sig"].flatten())
    true_change_time = exp_data["change"].flatten() - 1 # 0 indexing

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    signal_matrix, lick_matrix = create_vectorised_data(datapath, lick_exponential_decay=False, decay_constant=0.5)

    # get parameter with lowest training loss
    loss = training_result["loss"]
    epoch_num = onp.where(loss == min(loss))[0][0] # need to look through cost
    params = training_result["param_val"][epoch_num]
    num_non_hazard_rate_param = 2

    # get the posterior with only posterior-relevant params
    global transition_matrix_list
    transition_matrix_list = make_transition_matrix(params[num_non_hazard_rate_param:])
    vectorised_posterior_calc = vmap(forward_inference_custom_transition_matrix)
    posterior = vectorised_posterior_calc(signal_matrix)

    # use lick matrix to turn some of the posterior to NaNs
    masked_posterior = np.where(lick_matrix==99, onp.nan, posterior)

    fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    axs = axs.flatten()
    if random_seed is not None:
        onp.random.seed(random_seed)
    for n, change_val in enumerate(onp.unique(change_values)):
        change_val_index = onp.where(change_values == change_val)[0]
        axs[n].set_title("Change magnitude: " + str(change_val))
        for plot_index in onp.random.choice(change_val_index, num_examples):
            if plot_peri_stimulus is True:
                start_time = true_change_time[plot_index] - 20
                end_time = true_change_time[plot_index] + 20
                peri_stimulus_time = np.arange(-20, 20)
                axs[n].plot(peri_stimulus_time, masked_posterior[plot_index, start_time:end_time], alpha=0.8)
                if plot_cumulative is True:
                    cumulative_posterior = predict_cumulative_lick(None, masked_posterior[plot_index, :])
                    axs[n].plot(peri_stimulus_time, cumulative_posterior[start_time:end_time], alpha=0.8, label="cumulative P(lick)")
            else:
                axs[n].plot(masked_posterior[plot_index, :], alpha=0.8)
                if plot_cumulative is True:
                    cumulative_posterior = predict_cumulative_lick(None, masked_posterior[plot_index, :])
                    axs[n].plot(cumulative_posterior, alpha=0.8, label="cumulative P(lick)")

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


# Plot single model comparison with fitted mouse data


def plot_psychometric_individual_sample(mouse_df_path, model_df_path, metric="lick", plot_examples=False, ylabel="P(lick)",
                                        remove_FA=False, shade_statistic="std"):
    """

    :param mouse_df_path:
    :param model_df_path: must include individual samples (rather than averaged across simulations)
    :param metric to plot on y-axis. Usually: FA, correct_lick, lick
    :param figsavepath:
    :param shade_statistic: the statistic used in the shading around the line plot
        "std"        | shaded region is +/- 1 std from the mean/median
        "confidence" | shaded region is the 95% confidence interval around the mean or median
    :return:
    """

    mouse_df = pd.read_pickle(mouse_df_path)
    model_df = pd.read_pickle(model_df_path)

    # TODO: likely will be more elegant with dict mapping
    if metric == "lick":
        mouse_metric = "mouse_lick"
    elif metric == "correct_lick":
        mouse_metric = "correct_lick"
    elif metric == "FA":
        mouse_metric = "mouse_FA"
    elif metric == "miss":
        mouse_metric = "mouse_miss"
    elif metric == "hit_sub":
        mouse_metric = "hit_sub"


    # Randomly plot some samples of psychometric to check
    if plot_examples is True:
        random_sim_index = np.random.choice(np.unique(model_df["sample_ID"]), 9)
        fig, axs = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True)
        axs = axs.flatten()
        for n, sim_index in enumerate(random_sim_index):
            subset_index = np.where(model_df["sample_ID"] == sim_index)[0]
            df_subset = model_df.iloc[subset_index]
            df_prop_choice = df_subset.groupby(["change"], as_index=False).agg({metric: "mean"}) # mean here is actually used to get proportion
            axs[n].plot(np.exp(df_prop_choice["change"]), df_prop_choice[metric], label="Model")
            axs[n].set_title("Simulation: " + str(sim_index))
            axs[n].spines["top"].set_visible(False)
            axs[n].spines["right"].set_visible(False)

        # common x y labels (and grid lines if necessary
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Change magnitude")
        plt.ylabel("P (licks)")
        plt.show()

    ###### remove FA
    if remove_FA is True:
        # subset model df based on model FA
        model_no_FA_index = np.where(model_df["FA"] != 1)[0]
        model_df = model_df.iloc[model_no_FA_index]

        # subset model df based on mouse FA
        mouse_no_FA_index = np.where(mouse_df["mouse_FA"] != 1)[0]
        model_df = model_df.loc[model_df["trial_ID"].isin(mouse_no_FA_index)]


        mouse_no_FA_index = np.where(mouse_df["mouse_FA"] != 1)[0]
        mouse_df = mouse_df.iloc[mouse_no_FA_index]


    ###################### mean and std across simulations #####################

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    df_prop_choice_all_simulations = model_df.groupby(["change", "sample_ID"], as_index=False).agg({metric: "mean"})
    df_prop_choice = df_prop_choice_all_simulations.groupby(["change"], as_index=False).agg({metric: "mean"})
    df_std = df_prop_choice_all_simulations.groupby(["change"], as_index=False).agg({metric: "std"})
    df_sem = df_prop_choice_all_simulations.groupby(["change"], as_index=False).agg({metric: "sem"})
    df_prop_choice["change"] = np.exp(df_prop_choice["change"])


    # Model
    ax.plot(df_prop_choice["change"], df_prop_choice[metric], label="Model")
    if shade_statistic == "std":
        ax.fill_between(df_prop_choice["change"], df_prop_choice[metric] - df_std[metric],
                        df_prop_choice[metric] + df_std[metric], alpha=0.3)
    elif shade_statistic == "confidence":
        ax.fill_between(df_prop_choice["change"], df_prop_choice[metric] - df_sem[metric] * 1.96,
                        df_prop_choice[metric] + df_sem[metric] * 1.96, alpha=0.3)

    ax.scatter(df_prop_choice["change"], df_prop_choice[metric], label=None, color="blue")

    # Mouse
    mouse_prop_choice = mouse_df.groupby(["change"], as_index=False).agg({mouse_metric: "mean"})

    ax.plot(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[mouse_metric], label="Mouse")
    ax.scatter(np.exp(mouse_prop_choice["change"]), mouse_prop_choice[mouse_metric], label=None)

    ax.legend(frameon=False)

    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Change magnitude")
    ax.set_ylabel(ylabel)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig, ax


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

def plot_chronometric(mouse_df_path, model_df_path, summary_stat="median"):
    # TODO: psychometric plot can be generalised to incoporate this.
    """
    Plots median of peri-stimulus reaction time for Hits
    :param mouse_df_path:
    :param model_df_path:
    :param figsavepath:
    :return:
    """

    model_metric = "peri_stimulus_rt"
    mouse_metric = "peri_stimulus_rt"

    mouse_df = pd.read_pickle(mouse_df_path)
    model_df = pd.read_pickle(model_df_path)

    ##### Subset only HIT trials
    model_hit_index = np.where(model_df["correct_lick"] == 1)[0]
    mouse_hit_index = np.where(mouse_df["mouse_hit"] == 1)[0]

    model_df = model_df.iloc[model_hit_index]
    mouse_df = mouse_df.iloc[mouse_hit_index]

    model_pivot_df = model_df.groupby(["change"], as_index=False).agg({model_metric: summary_stat})
    mouse_pivot_df = mouse_df.groupby(["change"], as_index=False).agg({mouse_metric: summary_stat})


    model_pivot_df["change"] = np.exp(model_pivot_df["change"])
    mouse_pivot_df["change"] = np.exp(mouse_pivot_df["change"])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(model_pivot_df["change"], model_pivot_df["peri_stimulus_rt"], label="Model")
    ax.scatter(model_pivot_df["change"], model_pivot_df["peri_stimulus_rt"], label=None, color="blue")

    ax.plot(mouse_pivot_df["change"], mouse_pivot_df["peri_stimulus_rt"], label="Mouse")
    ax.scatter(mouse_pivot_df["change"], mouse_pivot_df["peri_stimulus_rt"], label=None, color="orange")

    model_pivot_df_sem = model_df.groupby(["change"], as_index=False).agg({model_metric: "sem"})
    ax.fill_between(model_pivot_df["change"], model_pivot_df["peri_stimulus_rt"] - 1.96 * model_pivot_df_sem[model_metric],
                    model_pivot_df["peri_stimulus_rt"] + 1.96 * model_pivot_df_sem[model_metric])

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel("Peri-change reaction time (frames)")

    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig, ax


def plot_psychometric_subjective(model_sample_path, mouse_beahviour_df_path=None, metric="prop_lick", label="Proportion of licks",
                      savepath=None, showfig=True):
    # TODO: This can be combined with plot_psychometric by specifying the groupby argument (separately for model and mouse)
    df = pd.read_pickle(model_sample_path)
    df_prop_choice = df.groupby(["subjective_change_value"], as_index=False).agg({metric: "mean"})
    df_std = df.groupby(["subjective_change_value"], as_index=False).agg({metric: "std"})

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_prop_choice["subjective_change_value"], df_prop_choice[metric], label="Model")
    ax.fill_between(df_prop_choice["subjective_change_value"], df_prop_choice[metric] - df_std[metric],
                    df_prop_choice[metric] + df_std[metric], alpha=0.3)
    ax.scatter(df_prop_choice["subjective_change_value"], df_prop_choice[metric], label=None, color="blue")

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
        mouse_prop_choice = mouse_df.groupby(["subjective_change_value"], as_index=False).agg({mouse_metric: "mean"})

        ax.plot(mouse_prop_choice["subjective_change_value"], mouse_prop_choice[mouse_metric], label="Mouse")
        ax.scatter(mouse_prop_choice["subjective_change_value"], mouse_prop_choice[mouse_metric], label=None)

        ax.legend(frameon=False)

    if "prop" in metric:
        ax.set_ylim([0, 1.05])

    ax.set_xlabel("Subjective change magnitude")
    ax.set_ylabel(label)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()


def compare_model_and_mouse_dist(mouse_df_path, model_df_path, plot_type="FA", change_magnitude=None, savepath=None,
                                 strip_dot_size=2, strip_jitter=1, strip_dot_alpha=1, model_sub_sample=1000,
                                 show_model_sub_sample=None, showfig=True):

    mouse_df = pd.read_pickle(mouse_df_path)
    model_df = pd.read_pickle(model_df_path)

    # TODO: just add the response as a feature in the df
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

    plt.rc('text', usetex=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.5, 0.5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    axs = [ax1, ax2, ax3]

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # remove tick marks
    axs[1].tick_params(length=0)
    axs[2].tick_params(axis="y", length=0)

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

def plot_multi_model_psychometric_error(model_comparison_df, multiple_mice=False,
                                        model_names=["Control", "Sigmoid decision"],
                                        metric="hit_exclude_FA_fit"):

    if multiple_mice is True:
        model_comparison_df_pivot = pd.pivot_table(model_comparison_df, index="model_number", columns="mouse_number",
                                                   values=metric)
        fig, ax = ts_boxplot(df=model_comparison_df_pivot)
        ax.set_ylabel("Mean-square error")
        ax.set_xlabel("Model")
        ax.legend(title="Mouse")
        plt.xticks(ticks=[1, 2], labels=model_names)
    else:
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

    return fig, ax


def plot_multi_model_rt_error(model_comparison_df, multiple_mice=False, model_names=["Control", "Sigmoid decision"],
                              rt_metric="FA_rt_dist_fit"):

    if multiple_mice is False:
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

    else:
        # compare multiple mice
        model_comparison_df_pivot = pd.pivot_table(model_comparison_df, index="model_number", columns="mouse_number",
                                                   values=rt_metric)
        fig, ax = ts_boxplot(df=model_comparison_df_pivot)
        ax.set_ylabel("KS statistic")
        ax.set_xlabel("Model")
        ax.legend(title="Mouse")
        plt.xticks(ticks=[1, 2], labels=model_names)

    return fig


def ts_boxplot(df):
    """
    My own custom boxplot function which includes lines connecting individuals.
    Note the strict format requirement of argument df
    :param df: pivoted dataframe, where each row is a factor, and each column is an individual
    :return:
    """
    plt.style.use("~/Dropbox/notes/Projects/second_rotation_project/normative_model/ts.mplstyle")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(x=df)
    ax.grid()

    num_factors = np.shape(df)[0]
    num_individuals = np.shape(df)[1]
    individual_number_list = np.arange(1, num_individuals+1)

    for n, individual_row in enumerate(df.values.T):
        ax.plot(np.arange(1, num_factors+1), individual_row, linewidth=1.0, alpha=0.2)  # plot individual data points
        ax.scatter(np.arange(1, num_factors+1), individual_row, alpha=0.8, edgecolors='none', s=10,
                   label=str(individual_number_list[n]))

    return fig, ax


# Plot experimental data / simple diagnostics

def plot_signals(datapath):
    """
    Plots the signal presented to the mouse.
    :param datapath:
    :return:
    """
    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signal = exp_data["ys"].flatten()
    random_index = baserandom.choice(np.arange(0, len(signal)))
    signal_sample = signal[random_index].flatten()

    # signal_sample = np.exp(signal_sample)
    signal_sample = np.power(2, signal_sample)

    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(signal_sample)
    plt.show()

    # TODO: Estimate the mean and variance from the data.

    print("Signal mean before change time:", np.mean(signal_sample[:tau[random_index]]))
    print("Signal mean after change time:", np.mean(signal_sample[tau[random_index]:]))


def plot_posterior(datapath, model_training_data_path=None, figsavepath=None):

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"]
    change_magnitude = exp_data["sig"]
    change_time = exp_data["change"]

    # Plot mean posterior over time grouped by stimulus magnitude

    posterior_list = list()

    global transition_matrix_list
    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath)

    for signal in tqdm(signals):
        print(np.shape(signal))
        posterior = forward_inference_custom_transition_matrix(signal[0].flatten())
        posterior_list.append(posterior)


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    windowed_posterior = list()
    window_length = 100
    for change_mag in np.unique(change_magnitude):
        change_index = onp.where(change_magnitude == change_mag)[0]
        for index in change_index:
            start_index = change_time[index] - window_length/2
            end_index = change_time[index] + window_length/2
            windowed_posterior.append(posterior_list[start_index:end_index])

        windowed_posterior_array = np.array(windowed_posterior)
        mean_posterior = np.mean(windowed_posterior, axis=0)

        ax.plot(mean_posterior, label=str(change_mag))

    ax.set_xlabel("Peri-change time (frames)")
    ax.set_ylabel(r"$P(z_k = \mathrm{change} \vert x_{1:k})$")

    if figsavepath is not None:
        plt.savefig(figsavepath)

    plt.show()


def plot_signal_and_inference(signal, tau, prob, savepath=None):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(signal)
    axs[0].axvline(tau, color="r", linestyle="--")

    axs[1].plot(prob)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    plt.show()


def plot_plick_examples(model_data_path, plot_peri_stimulus=True, num_examples=10, random_seed=None, figsavepath=None):
    """
    Plot some example of the
    :param model_data_path:
    :return:
    """

    with open(model_data_path, "rb") as handle:
        model_data = pkl.load(handle)

    fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharey=True, sharex=True)
    axs = axs.flatten()
    if random_seed is not None:
        np.random.seed(777)
    for n, change_val in enumerate(np.unique(model_data["change_value"])):
        change_val_index = np.where(model_data["change_value"] == change_val)[0]
        axs[n].set_title("Change magnitude: " + str(change_val))
        for plot_index in np.random.choice(change_val_index, num_examples):
            if plot_peri_stimulus is True:
                start_time = model_data["true_change_time"][plot_index] - 20
                end_time = model_data["true_change_time"][plot_index] + 20
                per_stimulus_time = np.arange(-20, 20)
                axs[n].plot(per_stimulus_time, model_data["model_vec_output"][plot_index][start_time:end_time], alpha=0.8)
            else:
                axs[n].plot(model_data["model_vec_output"][plot_index], alpha=0.8)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


# Helper functions


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _standard_sigmoid(input):
    # Naive version:
    # output = 1.0 / (1.0 + np.exp(-input))
    # Numerically stable version
    """
    if input >= 0:
        z = np.exp(-input)
        output = 1.0 / (1.0 + z)
    else:
        z = np.exp(input)
        output = z / (1.0 + z)
    """

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                    1.0 / (1.0 + np.exp(-input)),
                    np.exp(input) / (1.0 + np.exp(input)))


    return output

def _nonstandard_sigmoid(input, min_val=0.0, max_val=1.0, k=1, midpoint=0.5):
    # naive implementation
    # output = (max_val - min_val) / (1.0 + np.exp(-input)) + min_val

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                      (max_val - min_val) / (1.0 + np.exp(-(input - midpoint) * k)) + min_val,
                      (max_val - min_val) * np.exp((input - midpoint) * k) / (1.0 + np.exp((input - midpoint) * k))
                      + min_val)

    return output