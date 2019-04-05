import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os

# bocd (not v. well documnted)
import bocd
from functools import partial

# bcp (slightly better documented)
from bayesianchangepoint import bcp

def main():
    # define paths
    fig_folder = "/home/timothysit/Dropbox/notes/Projects/second_rotation_project/normative_model/figures/"


    # simulate stimulus, parameters from Orsolic 2019
    """
    On noisy trials, temporal frequency of the grating was drawn every 50 ms (3 monitor frames) from a
    lognormal distribution, such that log2-transformed TF had the mean of 0 and standard deviation of 0.25 octaves.
    """
    u_1 = 1.0
    u_2 = 1.25
    sigma = 0.25
    update_interval = 1
    n = 3000

    time_series = sim_data(u_1=u_1, u_2=u_2, sigma=sigma, tau=1500,
                           update_interval=update_interval, n=n, dist="log_normal")

    plot_data(time_series)

    # change_points = detect_change(time_series)
    # print(change_points)
    detect_change_bcp(time_series, figpath=os.path.join(fig_folder, "test2.png"),showfig=True)

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


    if update_interval > 1:
        signal_1 = np.repeat(signal_1, update_interval)
        signal_2 = np.repeat(signal_2, update_interval)

    signal = np.append(signal_1, signal_2)

    return signal




def plot_data(data):

    fig, ax = plt.subplots(figsize=(8, 6))

    samples = np.arange(1, len(data)+1)
    ax.plot(samples, data)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_ylabel("Temporal frequency")
    ax.set_xlabel("Samples")

    plt.show()


def detect_change_bocd(data):
    lamb = 100
    alpha = 0.1
    beta = 1.0
    kappa = 1.0
    mu = 0.0
    delay = 15
    threshold = 0.5

    model = bocd.BOCD(partial(bocd.constant_hazard, lamb),
                      bocd.StudentT(alpha, beta, kappa, mu))
    changepoints = []

    for n, point in enumerate(data):
        model.update(point)
        if model.growth_probs[n] >= threshold:
            changepoints.append(model.t - n + 1)

    return changepoints

def detect_change_bcp(data, figpath=None, showfig=True):

    hazard_func = lambda r: bcp.constant_hazard(r, _lambda=len(data)) # original: 200
    beliefs, maxes = bcp.inference(data, hazard_func)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))

    ax[0].plot(data)
    ax[1].imshow(np.log(beliefs), interpolation='none', aspect='auto',
                 origin='lower', cmap=plt.cm.Blues)
    ax[1].plot(maxes, color='r', linewidth=1, linestyle="--")
    ax[1].set_xlim([0, len(data)])
    ax[1].set_ylim([0, ax[1].get_ylim()[1]])
    ax[0].grid()
    ax[1].grid()
    index_changes = np.where(np.diff(maxes.T[0]) < 0)[0]
    ax[0].scatter(index_changes, data[index_changes], c='green')

    ax[0].set_ylabel("Temporal frequency")

    ax[1].set_ylabel("Run length")
    ax[1].set_xlabel("Samples")

    if figpath is not None:
        plt.savefig(figpath, dpi=300)

    if showfig is True:
        plt.show()


def plot_hazard_rate(lambda_val, num_samples=100):
    """
    Plot hazard rate for different lambda values.
    Just as an illustration.
    :return:
    """
    p_gap = []
    p_change = 1.0 / lambda_val
    for n_samp in np.arange(1, num_samples):
        p_gap.append((1 - p_change)**n_samp * p_change)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, num_samples), p_gap)

    ax.set_xlabel("gap")
    ax.set_ylabel("P(gap)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # TODO: Might also be useful to plot the cumulative probability

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(np.cumsum(p_gap))

    plt.show()








if __name__ == "__main__":
    main()
    # plot_hazard_rate(lambda_val=500.0, num_samples=1000)

