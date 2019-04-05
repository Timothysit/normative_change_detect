import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import os


def sim_data(u_1, u_2, sigma, tau, n, update_interval, dist="log_normal"):
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
        print("No valid conditions specified")

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




u_1 = 1.0
u_2 = 2.0
sigma = 1
tau = 500
n = 1000
update_interval = 1
dist = "normal"

signal = sim_data(u_1, u_2, sigma, tau, n, update_interval, dist=dist)
signal = signal.reshape(-1, 1)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
model.startprob_ = np.array([1.0, 0.0])
# specify left-right HMM
model.transmat_ = np.array([[0.999, 0.001],
						   [0.0, 1.0]])
model.means_ = np.array([[1], [2]])
model.covars_ = np.array([[1], [1]])

sample_from_model(model=model)

figfolder = "/home/timothysit/Dropbox/notes/Projects/second_rotation_project/normative_model/figures/"
figname = "gaussian_HMM_test_1.png"
fit_model(model, signal, fit_model_to_data=False, tau=tau,
          savepath=os.path.join(figfolder, figname))


