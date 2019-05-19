# All the plotting functions
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random as baserandom # for random choice of signals/posterior to plot

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


def plot_trained_hazard_rate(training_savepath, figsavepath, num_non_hazard_rate_param=2):
    """
    Plot fitted hazard rate.
    :param training_savepath:
    :param figsavepath:
    :param num_non_hazard_rate_param:
    :return:
    """
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    last_epoch_trained_param = training_result["param_val"][-1]
    hazard_rate = last_epoch_trained_param[num_non_hazard_rate_param:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(standard_sigmoid(hazard_rate))
    # ax.plot(softmax(hazard_rate))
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("P(change)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()

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



# Helper functions


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)