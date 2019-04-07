import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl
import scipy.io
import pomegranate as pom
from tqdm import tqdm
import pandas as pd

def read_exp_data(filepath):
    """
    Reads matlab file
    :param filepath:
    :return:
    """
    data = scipy.io.loadmat(filepath)

    return data

def plot_exp_signals(exp_data, plot_index=1):
    signals = exp_data["ys"].flatten()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(signals[plot_index].flatten())
    ax.set_xlabel("Samples")
    ax.set_ylabel("Temporal frequency")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.show()






def sim_data(u_1, u_2, sigma, tau, n, update_interval, dist="log_normal",
             nosie_mean=None, noise_sigma=None):
    """
    simulates time series data with some a change in mean after a certain time t
    :param u_1: mean from the first data point up to the change point
    :param u_2: mean from the change point up to the end of the data
    :param sigma: variance of the data
    :param tau: time of the change (in samples)
    :param n: number of data points to generate
    :param update_interval: how frequently to update the stimulus; 1 means updated every frame,
                            3 means updated every 3 frames
    :param dist: distribution in which the variables should be drawn from
    :return:
    """

    assert tau % update_interval == 0, print("First segment not divisible by the update interval")
    assert (n-tau) % update_interval == 0, print("Second segment not divisible by the update interval")

    if dist == "log_normal":
        signal_1 = np.random.lognormal(mean=u_1, sigma=sigma, size=int(tau / update_interval))
        signal_2 = np.random.lognormal(mean=u_2, sigma=sigma, size=int((n-tau) / update_interval))
    elif dist == "normal":
        signal_1 = np.random.normal(loc=u_1, scale=sigma, size=int(tau / update_interval))
        signal_2 = np.random.normal(loc=u_2, scale=sigma, size=int((n-tau) / update_interval))
    else:
        print("No valid distributions specified")

    if update_interval > 1:
        signal_1 = np.repeat(signal_1, update_interval)
        signal_2 = np.repeat(signal_2, update_interval)

    signal = np.append(signal_1, signal_2)

    return signal

def sample_from_model(model):
    x, z = model.sample(1000)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(x)
    axs[1].plot(z)
    fig.suptitle("HMM model generated data")

    plt.show()

def sample_from_pom_model(model, n_sample=1000):

    sample, path = model.sample(n_sample, path=True)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(sample)
    axs[1].plot(path)
    fig.suptitle("HMM model generated data")

    plt.show()



def fit_model(model, signal, fit_model_to_data=True, tau=None, savepath=None):
    if fit_model_to_data is True:
        model.fit(signal)
    predicted_state = model.predict(signal)
    predicted_prob = model.predict_proba(signal)

    # Plot simulated data and predictions (OFFLINE)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(signal)

    if tau is not None:
        axs[0].axvline(tau, color="r", linestyle="--")

    axs[1].plot(predicted_prob)
    axs[1].legend([r"$P(z_1 \vert x)$", r"$P(z_2 \vert x)$"], frameon=False)

    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)

    axs[1].set_xlabel("Samples")

    if savepath is not None:
        fig.savefig(savepath)

    plt.show()

    # ONLINE (should be the same as "offline")
    p = np.zeros((len(signal), 2))
    for n, x_t in enumerate(signal):
        p[n, :] = model.predict_proba(x_t.reshape(-1, 1))

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle("Online predictions")
    axs[0].plot(signal)
    if tau is not None:
        axs[0].axvline(tau, color="r", linestyle="--")

    axs[1].plot(predicted_prob)
    axs[1].legend([r"$P(z_1 \vert x)$", r"$P(z_2 \vert x)$"], frameon=False)

    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)

    axs[1].set_xlabel("Samples")

    plt.show()

def random_choice(lick_bias=0.5):
    """
    :param: lick_bias | the probability of licking
    :param dist:
    :return:
    """

    uniform_random = np.random.random()
    if uniform_random < lick_bias:
        lick_choice = 1.0
    else:
        lick_choice = 0.0

    return lick_choice



def posterior_to_decision(p_state, policy=["softmax", "epsilon_greedy"], values=[1, 1, 1, 1], epsilon=0.1):
    """
    Given the estimated posterior probability of being in the change state,
    output the decision made.

    :param: p_state | probability of being in the change state given observations (from 1:t)
    :param: values | array of 4 values, associated with the *magnitudes* of
                    1. the value of true positive 2. the value of true negative
                    3. the value of false positive 4. the value of false negative
            if None, then these are not taken into account when making the decision.
    :return: lick_choice | if 0, then no lick, if 1, then lick
    """

    lick_benefit = p_state * values[0]
    lick_cost = (1-p_state) * values[2]
    lick_val = lick_benefit - lick_cost

    no_lick_benefit = (1-p_state) * values[1]
    no_lick_cost = p_state * values[3]
    no_lick_val = no_lick_benefit - no_lick_cost

    lick_choice = np.array((lick_val > no_lick_val)).astype(float)

    if "epsilon_greedy" in policy:
        uniform_random = np.random.random() # uniform random from 0 to 1
        if uniform_random < epsilon:
            lick_choice = random_choice(lick_bias=0.5)

    return lick_choice


def hmm_learn_define_model():
    """
    define a Hidden Markov Model using hmmlearn
    seems to be limited to the normal distribution,
    not sure about implementing log-normal distributions
    :return:
    """
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
    model.startprob_ = np.array([1.0, 0.0])
    # specify left-right HMM
    model.transmat_ = np.array([[0.999, 0.001],
                                [0.0, 1.0]])
    model.means_ = np.array([[1], [2]])
    model.covars_ = np.array([[1], [1]])

    return model

def pomegranate_define_model(u_baseline=0.0, u_change=1.0,
                             var_baseline=1.0, var_change=1.0):
    """
    define a Hidden Markov model using pomegranate
    has more flexibility of distributions compared to hmmlearn
    :return:
    """
    dists = [pom.LogNormalDistribution(u_baseline, var_baseline),
             pom.LogNormalDistribution(u_change, var_change)]
    trans_mat = np.array([[0.999, 0.001],
                           [0.0, 1.0]])
    starts = np.array([1.0, 0.0])

    model = pom.HiddenMarkovModel.from_matrix(trans_mat, dists, starts)

    return model


def predict_from_model(model, signal, tau=None, algorithm="map", savepath=None, plot_data=True):

    predictions = np.zeros(len(signal))
    posterior = np.zeros((len(signal), 2))
    for n_t in np.arange(1, len(signal)):
        prediction = model.predict(signal[0:n_t], algorithm=algorithm)
        predictions[n_t] = prediction[-1]
        p = model.predict_proba(signal[0:n_t])
        posterior[n_t, :] = p[-1, :]




    if plot_data is True:
        # Plot simulated data and predictions (OFFLINE)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(np.log(signal))
        axs[0].set_ylabel("Temporal frequency (Hz)")

        if tau is not None:
            axs[0].axvline(tau, color="r", linestyle="--")

        # axs[1].plot(prediction)
        # axs[1].legend([r"$P(z_1 \vert x)$", r"$P(z_2 \vert x)$"], frameon=False)
        axs[1].plot(posterior[:, 1])
        axs[1].set_ylabel(r"$P(z_{2, n} \vert x_{1:n})$")

        axs[0].spines["right"].set_visible(False)
        axs[0].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["top"].set_visible(False)

        axs[1].set_xlabel("Samples")

        if savepath is not None:
            fig.savefig(savepath)

        plt.show()

    return posterior



def simulate_trial(u_1, u_changes, sigma, tau, n, update_interval, model, n_trial=1000):
    """
    Simulate change detection trials
    :param u_1: baseline mean
    :param u_changes: list of mean of the change state
    :param sigma:
    :param tau:
    :param n:
    :param update_interval:
    :param model:
    :param n_trial:
    :return:
    """

    change = list()
    decision = list()
    decision_time = list()


    for trial in tqdm(np.arange(0, n_trial)):
        u_2 = np.random.choice(u_changes)
        signal = sim_data(u_1, u_2, sigma, tau, n, update_interval, dist="log_normal")
        signal = signal.reshape(-1, 1)

        posterior = predict_from_model(model, signal, tau=tau, algorithm="map",
                                       savepath=None, plot_data=False)

        lick_choice = posterior_to_decision(posterior[:, 1], policy=["softmax", "epsilon_greedy"], values=[1, 1, 1, 1],
                                            epsilon=0.1)

        # save data
        change.append(u_2)
        decision.append(np.max(lick_choice)) # 0 = no lick, 1 = lick
        if np.max(lick_choice) == 1:
            decision_time.append(np.where(lick_choice == 1)[0][0])
        else:
            decision_time.append(np.nan)


    simulated_data = dict()
    simulated_data["change"] = change
    simulated_data["decision"] = decision
    simulated_data["decision_time"] = decision_time


    return simulated_data


def example_run():
    u_1 = 1.0
    u_2 = 1.25
    u_changes = [1.00, 1.25, 1.35, 1.50, 2.00, 4.00]
    sigma = 0.25
    tau = 500
    n = 1000
    update_interval = 1
    dist = "log_normal"

    signal = sim_data(u_1, u_2, sigma, tau, n, update_interval, dist=dist)
    signal = signal.reshape(-1, 1)

    # model = hmm_learn_define_model()
    # sample_from_model(model=model)

    model = pomegranate_define_model(u_baseline=u_1, u_change=np.mean(u_changes), var_baseline=0.25, var_change=0.25)

    figfolder = "/home/timothysit/Dropbox/notes/Projects/second_rotation_project/normative_model/figures/"
    figname = "log_normal_HMM_test_1.png"

    posterior = predict_from_model(model, signal, tau=tau, algorithm="map",
                                   savepath=os.path.join(figfolder, figname), plot_data=True)

    lick_choice = posterior_to_decision(posterior[:, 1], policy=["softmax", "epsilon_greedy"], values=[1, 1, 1, 1], epsilon=0.1)

    plt.plot(lick_choice)
    plt.show()

    # fit_model(model, signal, fit_model_to_data=False, tau=tau,
    #           savepath=os.path.join(figfolder, figname))

    exp_file_path = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/data/data_IO_083.mat"

def plot_psychometric_curve(df, metric="decision", label="Proportion lick", savepath=None, showfig=True):


    df_prop_choice = df.groupby(["change"], as_index=False).agg({metric: "mean"})

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df_prop_choice.change, df_prop_choice.decision)
    ax.scatter(df_prop_choice.change, df_prop_choice.decision)

    if metric == "decision":
        ax.set_ylim([0, 1])

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel(label)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath)

    if showfig is True:
        plt.show()


def main(simulate_data=False, plot_data=False):
    if simulate_data is True:
        # example_run()
        u_1 = 1.0
        u_changes = [1.00, 1.25, 1.35, 1.50, 2.00, 4.00]
        sigma = 0.25
        tau = 500
        update_interval = 1
        n = 1000
        n_trial = 100
        model = pomegranate_define_model(u_baseline=u_1, u_change=np.mean(u_changes), var_baseline=0.25, var_change=0.25)

        simulated_data = simulate_trial(u_1=u_1, u_changes=u_changes, sigma=sigma, model=model,
                                        update_interval=update_interval, tau=tau, n=n, n_trial=n_trial)

        simulated_data = pd.DataFrame.from_dict(simulated_data)

        df_file_name = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/test.pkl"
        with open(df_file_name, "wb") as handle:
            pkl.dump(simulated_data, handle)
    if plot_data is True:
        df_file_name = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/test.pkl"
        df = pd.read_pickle(df_file_name)
        plot_psychometric_curve(df, metric="decision", label="Proportion lick", savepath=None, showfig=True)





if __name__ == "__main__":
    main(simulate_data=False,
         plot_data=True)
