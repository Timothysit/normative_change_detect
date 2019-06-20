# Automatic Differentiation using jax
# https://github.com/google/jax
import jax.numpy as np
import numpy as onp  # original numpy for indexed assignment/mutation (outside context of differentiation)
import numpy.random as npr # randomisation for minibatch gradient descent
from jax import grad, jit, vmap
from jax.experimental import optimizers
import jax.random as random
import jax.lax as lax

from jax import device_put  # onp operations not transferred to GPU, so they should run faster

# Other things
import hmm  # functions to simulate data
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import itertools

import os
import pickle as pkl
import pandas as pd
from os.path import expanduser

# smoothing (hazard rate regularisation)
import smoothing

from scipy.signal import savgol_filter

# debugging nans returned by grad
from jax.config import config

# Cross validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Plotting functions
import normative_plot as nmt_plot
import functools  # to pass around sigmoid functions for plotting
# stylesheet_path = "https://github.com/Timothysit/normative_change_detect/blob/master/ts.mplstyle"
stylesheet_path = "ts.mplstyle"

def cal_p_x_given_z(x_k):
    z_mu = np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00]))
    z_var = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2  # in actual data, std is 0.25

    # convert standard deviation from normal distribution to log-e normal
    # z_var_normal = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2
    # z_var = np.exp(z_var_normal) * (np.exp(z_var_normal) - 1)  # should be around 0.0687, and so std = 0.262

    # use log-e normal mu
    # z_mu = np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00])

    p_x_given_z = (1 / np.sqrt(2 * np.pi * z_var)) * np.exp(-(x_k - z_mu) ** 2 / (2 * z_var))

    # returns row vector (for simpler calculation later on)
    return p_x_given_z.T


def forward_inference(x):
    # rewritten to use without assignment of the form: A[i, j] = x
    # TODO: Think about logging thing to prevent underflow
    p_z1_given_x = list()
    p_z2_given_x = list()

    hazard_rate = 0.0001

    transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0,
                                   hazard_rate/5.0],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])
    init_state_probability = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # transition_matrix = np.array([[1 - hazard_rate, hazard_rate],
    #                               [0, 1]])

    # List to store the posterior: p(z_k | x_{1:k}) for k = 1, ..., n
    p_z_given_x = np.zeros((len(x), 6)) # this will be a n x M matrix, not sure if this is can be created without array assignment..
    p_change_given_x = list()
    p_baseline_given_x = list()
    p_xk_given_z_store = np.zeros((len(x), 6))

    # Initial probabilities
    p_z_and_x = cal_p_x_given_z(x_k=x[0]) * init_state_probability

    # add the initial probability to the output list
    p_change_given_x.append(0)


    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_xk_given_z = cal_p_x_given_z(x_k=x[k])


        # update joint
        p_z_and_x = np.dot((p_xk_given_z * p_z_and_x), transition_matrix)

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_zk_given_xk = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_zk_given_xk[1:])) # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    # return p_z_given_x
    return np.array(p_change_given_x)
    # return p_xk_given_z_store


def forward_inference_w_tricks(x):
    """
    Rewritten with filtering recursion.
    (Log-sum-exp won't work due to some transition probability values being zero)
    (Actually, it may not work because I can't take the log of the right to left transition probabilities...)
    :param x:
    :return:
    """
    p_z1_given_x = list()
    p_z2_given_x = list()

    hazard_rate = 0.0001

    transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])
    init_state_probability = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # transition_matrix = np.array([[1 - hazard_rate, hazard_rate],
    #                               [0, 1]])



    # List to store the posterior: p(z_k | x_{1:k}) for k = 1, ..., n
    p_z_given_x = np.zeros((len(x), 6)) # this will be a n x M matrix, not sure if this is can be created without array assignment..
    p_change_given_x = list()
    p_baseline_given_x = list()
    # p_xk_given_z_store = np.zeros((len(x), 6))

    # Initial probabilities
    p_z_and_x = cal_p_x_given_z(x_k=x[0]) * init_state_probability
    p_z_given_x = p_z_and_x / np.sum(p_z_and_x)


    # add the initial probability to the output list
    p_change_given_x.append(0)


    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_xk_given_z = cal_p_x_given_z(x_k=x[k])

        # update conditional probability
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix)

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:])) # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    # return p_z_given_x
    return np.array(p_change_given_x)
    # return p_xk_given_z_store

def forward_inference_custom_transition_matrix(x):
    """
    :param transtiion_matrix_list: global variable with list of transition matrices
    :param x: signal to predict
    :return:
    """

    p_z1_given_x = list()
    p_z2_given_x = list()

    init_state_probability = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # List to store the posterior: p(z_k | x_{1:k}) for k = 1, ..., n
    p_z_given_x = np.zeros((len(x), 6)) # this will be a n x M matrix,
    # not sure if this is can be created without array assignment..
    p_change_given_x = list()
    p_baseline_given_x = list()
    # p_xk_given_z_store = np.zeros((len(x), 6))

    # Initial probabilities
    p_z_and_x = cal_p_x_given_z(x_k=x[0]) * init_state_probability
    p_z_given_x = p_z_and_x / np.sum(p_z_and_x)


    # add the initial probability to the output list
    p_change_given_x.append(0)

    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_xk_given_z = cal_p_x_given_z(x_k=x[k])

        # update conditional probability
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix_list[k-1])

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:])) # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    return np.array(p_change_given_x)



def forward_inference_stat(x):
    """
    :param transtiion_matrix_list: global variable with list of transition matrices
    :param x: signal to predict
    :return:
    """

    p_change_given_z = list()

    init_state_probability = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # List to store the posterior: p(z_k | x_{1:k}) for k = 1, ..., n
    p_z_given_x = np.zeros((len(x), 6)) # this will be a n x M matrix,
    # not sure if this is can be created without array assignment..
    p_change_given_x = list()
    p_baseline_given_x = list()
    # p_xk_given_z_store = np.zeros((len(x), 6))

    # Initial probabilities
    p_z_and_x = cal_p_x_given_z(x_k=x[0]) * init_state_probability
    p_z_given_x = p_z_and_x / np.sum(p_z_and_x)


    # add the initial probability to the output list
    p_change_given_x.append(0)

    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_xk_given_z = cal_p_x_given_z(x_k=x[k])

        # update conditional probability
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix_list[k-1])

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:])) # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

        p_change_given_z.append(np.sum(p_change_given_z[1:]))

    return np.array(p_change_given_x)


def apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=1.0, false_negative=1.0, false_positive=1.0):
    """
    computes decision value based on cost and benefit
    note that true-positive is always set to 1.0, everything else is relative to it.
    some restrictions on parameter space:
         - (soft restriction) each weight value must be greater than 0
         - (hard restriction) sum of all weights must be greater than 0
    :param true_positive: relative reward for getting a Hit (lick when there is a change)
    :param true_negative: relative reward for a correct rejection (not licking when there is no change)
    :return:
    """
    # ensure parameter constraints are met
    small_value = 1e-9
    true_negative = np.max([small_value, true_negative])
    false_negative = np.max([small_value, false_negative])
    false_positive = np.max([small_value, false_positive])

    # cost benefit calculation
    lick_benefit = change_posterior * true_positive
    lick_cost = (1 - change_posterior) * false_positive
    no_lick_benefit = (1 - change_posterior) * true_negative
    no_lick_cost = change_posterior * false_negative

    lick_value = lick_benefit - lick_cost
    no_lick_value = no_lick_benefit - no_lick_cost

    prob_lick = (lick_value - no_lick_value) / (true_positive + true_negative + false_negative + false_positive)

    return prob_lick


def apply_strategy(prob_lick, k=10, midpoint=0.5, max_val=1.0, min_val=0.0,
                   policy="sigmoid", epsilon=0.1, lick_bias=0.5):
    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    if policy == "sigmoid":
        # two param logistic function
        p_lick = (max_val - min_val) / (1 + np.exp(-k * (prob_lick - midpoint))) + min_val
    elif policy == "epsilon-greedy":
        # epsilon-greedy policy
        p_lick = prob_lick * (1 - epsilon) + epsilon * lick_bias

    return p_lick

def apply_softmax_decision_rule(lick_value, no_lick_value, beta):
    """
    Applies softmax decision rule
    :param lick_value:
    :param no_lick_value:
    :param beta: inverse temperature (sometimes denoted as gamma in the literature)
    :return:
    """

    p_lick = 1 / (1 + np.exp(-beta * (lick_value - no_lick_value)))

    return p_lick


# Test forward algorithm
def test_forward_algorithm(tau=50):
    x = hmm.sim_data(u_1=0.0, u_2=1.0, sigma=0.25, tau=tau, n=500, update_interval=1, dist="normal",
                     noise_mean=None, noise_sigma=None, noise_dist="log_normal")

    # joint_prob = forward_inference(x)
    cond_prob = forward_inference_w_tricks(x)
    # cond_prob =forward_inference(x)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(x)
    axs[0].axvline(tau, color='r', linestyle="--")
    # axs[1].plot(joint_prob[:, 1])
    axs[1].plot(cond_prob)

    plt.show()


def test_on_data(exp_data_file, change_val=1.0):

    with open(exp_data_file, "rb") as handle:
        exp_data = pkl.load(handle)

    trial_type_list = exp_data["hazard"]
    change_magnitude = np.exp(exp_data["sig"].flatten())
    # noiseless_trial_type = exp_data["noiseless"].flatten()
    # mouse_abort = (exp_data["outcome"].flatten() == "abort").astype(float)

    # remove aborted trials, and noisless trials
    if change_magnitude is not None:
        # trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0) * (change_magnitude == change_val))[0]
        trial_index = onp.where(change_magnitude == change_val)[0]
    else:
        trial_index = onp.where((mouse_abort == 0) & (noiseless_trial_type == 0))[0]

    signal = exp_data["ys"].flatten()[trial_index][0][0]
    tau = exp_data["change"][trial_index][0][0]

    # use custom transition matrix
    global transition_matrix_list
    hazard_rate, transition_matrix_list = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_file)
    p_z_given_x = forward_inference_custom_transition_matrix(signal)

    global num_non_hazard_rate_params
    num_non_hazard_rate_params = 3

    param_vals = np.array([10.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    param_vals = np.hstack([param_vals, hazard_rate])
    p_lick = predict_lick_w_hazard_rate(param_vals=param_vals, signal=signal)

    # Plot example
    plot_signal_and_inference(signal=signal, tau=tau, prob=p_z_given_x)


def posterior_to_decision(posterior, return_prob=True, k=1.0, false_negative=0.0, midpoint=0.5):
    change_posterior = posterior  # p(z_2 | x_{1:k})
    p_lick = apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=false_negative, false_positive=0.0)
    p_lick = apply_strategy(p_lick, k=k, midpoint=midpoint)

    if return_prob is True:
        return p_lick


def gradient_descent_fit_vector(exp_data_path, training_savepath, init_param_vals=np.array([10.0, 0.0, 0.5]),
                                time_shift_list=np.arange(0, 5), num_epoch=10):
    """

    :return:
    """

    with open(exp_data_path, "rb") as handle:
        exp_data = pkl.load(handle)

    reaction_time = exp_data["rt"]
    mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()
    mouse_rt = exp_data["rt"].flatten()
    signal = exp_data["ys"].flatten()

    mouse_lick_vector_list = list()
    for trial in onp.arange(0, len(mouse_rt)):
        if not onp.isnan(mouse_rt[trial]):
            mouse_lick_vector = onp.zeros(shape=(int(mouse_rt[trial]), ))
            mouse_lick_vector[int(mouse_rt[trial] - 1)] = 1
            mouse_lick_vector_list.append(mouse_lick_vector)
        else:
            mouse_lick_vector = onp.zeros(shape=(len(signal[trial][0])), )
            mouse_lick_vector_list.append(mouse_lick_vector)

    assert len(signal) == len(mouse_lick_vector_list) # check equal number of trials

    signal = exp_data["ys"].flatten()

    # Define global variables used by loss_function
    # global actual_lick_vector
    # global signals
    actual_lick_vector = mouse_lick_vector_list
    signals = signal

    global time_shift

    loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))
    final_param_list = list() # list of list, each list is the parameter values for a particular time shift

    loss_val_list = list()

    for time_shift in time_shift_list:

        step_size = 0.0001
        momentum_mass = 0.9
        opt_init, opt_update = optimizers.momentum(step_size, mass=momentum_mass)

        @jit
        def update(i, opt_state):
            params = optimizers.get_params(opt_state)
            return opt_update(i, grad(loss_function_fit_vector_faster)(params), opt_state)

        opt_state = opt_init(init_param_vals)

        # TODO: think about doing batch gradient descent
        itercount = itertools.count()
        for epoch in range(num_epoch):
            opt_state = update(next(itercount), opt_state)
            params = optimizers.get_params(opt_state)
            loss_val = loss_function_fit_vector_faster(params)
            loss_val_list.append(loss_val)
            print("Loss:", loss_val)

        final_param_list.append(params)

    training_result = dict()
    training_result["loss"] = loss_val_matrix
    training_result["param_val"] = final_param_list

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def create_vectorised_data(exp_data_path, subset_trials=None, lick_exponential_decay=False, decay_constant=0.5,
                           full_trial_signal=False):
    """
    Create a matrix from signal and lick vectors of unequal length so that computations can be vectorised.
    :param exp_data_path:
    :param subset_trials:
    :param lick_exponential_decay: If true, then lick vector will be an exponential decay function starting
    at the point of lick and decaying backwards in time. This may help the model fit better.
    :return:
    """
    # LOADING DATA
    with open(exp_data_path, "rb") as handle:
        exp_data = pkl.load(handle)

    mouse_rt = exp_data["rt"].flatten()
    signal = exp_data["ys"].flatten()
    change_magnitude = exp_data["sig"].flatten()

    # CONVERTING VECTORS TO A PADDED MATRIX SO THINGS CAN BE VECTORISED
    # reshape signal to a matrix (so we can do vectorised operations on it)
    num_time_samples = list()
    for s in signal:
        num_time_samples.append(len(s[0]))
    max_time_bin = onp.max(num_time_samples)
    num_trial = len(signal)

    # signal_matrix = onp.zeros(shape=(num_trial, max_time_bin))

    # pad with random valuesrather than 0s, might help up underflow...
    # key = random.PRNGKey(0)
    # signal_matrix = random.normal(key, shape=(num_trial, max_time_bin))
    # signal_matrix = onp.random.normal(loc=0.0, scale=1.0, size=(num_trial, max_time_bin))
    signal_matrix = onp.zeros(shape=(num_trial, max_time_bin))

    for n, s in enumerate(signal):
        signal_matrix[n, :len(s[0])] = s[0] # up to full length of the signal in the trial
        signal_matrix[n, len(s[0]):] = onp.random.normal(loc=change_magnitude[n], scale=0.0625,
                                                         size=(max_time_bin - len(s[0])))

    lick_matrix = onp.zeros(shape=(num_trial, max_time_bin))
    lick_matrix[:] = 99

    # fill the matrix with ones and zeros
    for trial in np.arange(0, len(mouse_rt)):
        if not onp.isnan(mouse_rt[trial]):
            mouse_lick_vector = onp.zeros(shape=(int(mouse_rt[trial]), ))
            mouse_lick_vector[int(mouse_rt[trial] - 1)] = 1 # note the zero-indexing
            if lick_exponential_decay is True:
                # TODO: Plot this in a separate block, there is wasted computational time upstairs.
                time_before_lick = np.arange(0, int(mouse_rt[trial] - 1))
                exp_lick = np.exp(-decay_constant * time_before_lick)
                mouse_lick_vector = np.flipud(exp_lick)
        else:
            mouse_lick_vector = onp.zeros(shape=(len(signal[trial][0])), )
        lick_matrix[trial, :len(mouse_lick_vector)] = mouse_lick_vector

    return signal_matrix, lick_matrix


def create_vectorised_data_new(exp_data, subset_trials=None, lick_exponential_decay=False, decay_constant=0.5,
                           full_trial_signal=False):
    """
    Create a matrix from signal and lick vectors of unequal length so that computations can be vectorised.
    :param exp_data_path:
    :param subset_trials:
    :param lick_exponential_decay: If true, then lick vector will be an exponential decay function starting
    at the point of lick and decaying backwards in time. This may help the model fit better.
    :return:
    """

    mouse_rt = exp_data["rt"].flatten()
    signal = exp_data["ys"].flatten()
    change_magnitude = exp_data["sig"].flatten()

    # CONVERTING VECTORS TO A PADDED MATRIX SO THINGS CAN BE VECTORISED
    # reshape signal to a matrix (so we can do vectorised operations on it)
    num_time_samples = list()
    for s in signal:
        num_time_samples.append(len(s[0]))
    max_time_bin = onp.max(num_time_samples)
    num_trial = len(signal)

    # signal_matrix = onp.zeros(shape=(num_trial, max_time_bin))

    # pad with random valuesrather than 0s, might help up underflow...
    # key = random.PRNGKey(0)
    # signal_matrix = random.normal(key, shape=(num_trial, max_time_bin))
    # signal_matrix = onp.random.normal(loc=0.0, scale=1.0, size=(num_trial, max_time_bin))
    signal_matrix = onp.zeros(shape=(num_trial, max_time_bin))

    for n, s in enumerate(signal):
        signal_matrix[n, :len(s[0])] = s[0] # up to full length of the signal in the trial
        signal_matrix[n, len(s[0]):] = onp.random.normal(loc=change_magnitude[n], scale=0.0625,
                                                         size=(max_time_bin - len(s[0])))

    lick_matrix = onp.zeros(shape=(num_trial, max_time_bin))
    lick_matrix[:] = 99

    # fill the matrix with ones and zeros
    for trial in np.arange(0, len(mouse_rt)):
        if not onp.isnan(mouse_rt[trial]):
            mouse_lick_vector = onp.zeros(shape=(int(mouse_rt[trial]), ))
            mouse_lick_vector[int(mouse_rt[trial] - 1)] = 1 # note the zero-indexing
            if lick_exponential_decay is True:
                # TODO: Plot this in a separate block, there is wasted computational time upstairs.
                time_before_lick = np.arange(0, int(mouse_rt[trial] - 1))
                exp_lick = np.exp(-decay_constant * time_before_lick)
                mouse_lick_vector = np.flipud(exp_lick)
        else:
            mouse_lick_vector = onp.zeros(shape=(len(signal[trial][0])), )
        lick_matrix[trial, :len(mouse_lick_vector)] = mouse_lick_vector

    return signal_matrix, lick_matrix


def gradient_descent_fit_vector_faster(exp_data_path, training_savepath, init_param_vals=np.array([10.0, 0.0, 0.5]),
                                time_shift_list=np.arange(0, 5), num_epoch=10):
    """

    :return:
    """
    # Define global variables used by loss_function
    global signal_matrix
    global lick_matrix
    global transition_matrix_list
    # signal_matrix, lick_matrix = create_vectorised_data(exp_data_path)

    # USING EXPONENTIAL DECAY
    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path,
                                                        lick_exponential_decay=True, decay_constant=0.5)

    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_path)

    global time_shift
    # TODO: Think about how to do time shift in this vectorised version

    # loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))
    param_list = list() # list of list, each list is the parameter values for a particular time shift
    loss_val_list = list()
    # define batched prediction function using vmap
    global batched_predict_lick
    batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    # print("Initial parameters:", init_param_vals)

    config.update("jax_debug_nans", True) # nan debugging
    # TURN THIS OFF IF NOT DEBUGGING (CAUSES SLOWDOWNS)

    for time_shift in tqdm(time_shift_list):
        print("Time shift: ", str(time_shift))

        step_size = 0.0001 # originally 0.01
        momentum_mass = 0.4
        opt_init, opt_update = optimizers.momentum(step_size, mass=momentum_mass)
        # opt_init, opt_update = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-8)

        """
        @jit
        def update(i, opt_state):
            params = optimizers.get_params(opt_state)
            return opt_update(i, grad(loss_function_fit_vector_faster)(params), opt_state)
        """

        itercount = itertools.count()
        """
        for epoch in range(num_epoch):
            opt_state = update(next(itercount), opt_state)
            params = optimizers.get_params(opt_state)
            loss_val = loss_function_fit_vector_faster(params)
            print("Loss:", loss_val)
            print("Parameters:", params)
            param_list.append(params)
            loss_val_list.append(loss_val)
        """

        # Minibatch gradient descent

        num_train = onp.shape(signal_matrix)[0]
        batch_size = 128
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield signal_matrix[batch_idx, :], lick_matrix[batch_idx, :]
        batches = data_stream()

        opt_state = opt_init(init_param_vals)

        @jit
        def step(i, opt_state, batch):
            params = optimizers.get_params(opt_state)
            # print("Params within step:", params)
            # g = grad(loss_function_fit_vector_faster)(params)
            g = grad(loss_function_batch)(params, batch)
            return opt_update(i, g, opt_state)

        for epoch in range(num_epoch):
            for _ in range(num_batches):
                opt_state = step(epoch, opt_state, next(batches))

            if epoch % 10 == 0:
                params = optimizers.get_params(opt_state)
                print("Parameters", params)
                # loss_val = loss_function_fit_vector_faster(params)
                loss_val = loss_function_batch(params, (signal_matrix, lick_matrix))
                print("Loss:", loss_val)
                loss_val_list.append(loss_val)
                param_list.append(params)
                # print("Parameters:", params)

                # TODO: Write function to auto-stop on convergence

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = param_list

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def predict_lick(param_vals, signal):
    # NOTE: This function also depends on the global parameter "time_shift"

    # impose some hard boundaries

    # posterior = forward_inference_w_tricks(signal.flatten())

    key = random.PRNGKey(777)
    # signal = signal.flatten() + (random.normal(key, (len(signal.flatten()), )) * standard_sigmoid(param_vals[2]))
    signal = signal.flatten() + (random.normal(key, (len(signal.flatten()),)) * standard_sigmoid(param_vals[0]))
    # multiply standard normal by constant: aX + b = N(au + b, a^2\sigma^2)
    posterior = forward_inference_custom_transition_matrix(signal)

    # prevent overflow/underflow
    small_value = 1e-6
    posterior = np.where(posterior >= 1.0, 1-small_value, posterior)
    posterior = np.where(posterior <= 0.0, small_value, posterior)

    # posterior = forward_inference_custom_transition_matrix(signal.flatten())

    # p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
    #                             false_negative=0.0, false_positive=0.0) # param_vals[1]
    # p_lick = apply_strategy(posterior, k=param_vals[0], midpoint=param_vals[1])

    # p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
    #                             false_negative=0.0, false_positive=0.0) # param_vals[1]
    # p_lick = apply_strategy(p_lick, k=nonstandard_sigmoid(param_vals[0], min_val=0, max_val=20),
    #                         midpoint=nonstandard_sigmoid(param_vals[1], min_val=-10, max_val=10),
    #                         max_val=1-1e-5, min_val=1e-5)
    p_lick = posterior

    # Add time shift

    if time_shift > 0:
        baseline = 0.88888888 # nan placeholder
        p_lick = np.concatenate([np.repeat(baseline, time_shift), p_lick])
        p_lick = p_lick[0:len(signal.flatten())]  # clip p(lick) to be same length as actual lick
    elif time_shift < 0:
        # backward shift
        baseline = 0.88888888  # nan placeholder
        p_lick = np.concatenate([p_lick, np.repeat(baseline, abs(time_shift))])
        p_lick = p_lick[abs(time_shift):(len(signal.flatten()) + abs(time_shift))]

    return p_lick


def loss_function_fit_vector_faster(param_vals):
    # missing arguments: signals, param_vals_, actual_lick_,
    """
    loss between predicted licks and actual licks
    computed using cross entropy
    This loss function is for licking versus no-licking, but does not compute the loss of the reaction time.
    :param: param_vals, array of parameter values, in the following order:
        1. k parameter in the sigmoid function
        2. false_negative value in cost-benefit function
        3. midpoint of the sigmoid decision function
    :param: signal, batched signal_matrix (ie. a subset of signal_matrix)
    :return:
    """

    batch_predictions = batched_predict_lick(param_vals, signal_matrix)
    # batch_loss = batched_cross_entropy(actual_lick_vector=lick_matrix, p_lick=batch_predictions)
    batch_loss = matrix_cross_entropy_loss(lick_matrix, batch_predictions)

    return batch_loss


def loss_function_batch(param_vals, batch, smoothing_lambda):
    # missing arguments: signals, param_vals_, actual_lick_,
    """
    loss between predicted licks and actual licks
    computed using cross entropy.
    This version computes the loss for a batch.

    Note that loss is now always a positive value (this seems to be the assumed input of JAX optimizers, in contrast to
    autograd optimisers, which seems to take negative loss values).

    """
    signal, lick = batch

    batch_predictions = batched_predict_lick(param_vals, signal)
    # batch_loss = batched_cross_entropy(actual_lick_vector=lick_matrix, p_lick=batch_predictions)
    # batch_loss = matrix_cross_entropy_loss(lick, batch_predictions)
    batch_loss = matrix_weighted_cross_entropy_loss(lick, batch_predictions, alpha=1, cumulative_lick=False)

    # adding log barrier
    # c = 10
    # barrier_loss = - c * np.sum(np.log(param_vals[num_non_hazard_rate_params:])) - c * np.sum(np.log(1 - param_vals[num_non_hazard_rate_params:]))
    # batch_loss = batch_loss + barrier_loss
    # print("Barrier loss: ", str(barrier_loss))

    # smoothing penalty for hazard rate (0 by default to have no penalty)
    batch_loss = batch_loss + smoothing.second_derivative_penalty(param_vals[num_non_hazard_rate_params:],
                                                           lambda_weight=smoothing_lambda)

    return batch_loss


def performance_batch(param_vals, batch):
    """
    Same as loss_function_batch, but also calculates other model peformance measures (accuracy, recall)
    :param param_vals:
    :param batch:
    :return:
    """
    signal, lick = batch
    batch_predictions = batched_predict_lick(param_vals, signal)
    batch_loss = matrix_cross_entropy_loss(lick, batch_predictions)

    # TODO: overall accuracy (regardless of time)
    # This may actually require sampling from the model...



def cross_entropy_loss(actual_lick_vector, p_lick):
    # find last actual value of the lick vector (effectively like a mask)
    # Can't use single argument np.where in JAX
    # first_invalid_value = np.where(actual_lick_vector == 99, 1, 0)
    # actual_lick_vector = actual_lick_vector[:first_invalid_value]
    # p_lick = p_lick[:first_invalid_value]

    # Can't use boolean indexing in JAX
    # actual_lick = actual_lick_vector[actual_lick_vector != 99]
    # p_lick = p_lick[actual_lick_vector != 99]

    # dot product indexing that uses 3 argument np.where
    # index_values = np.where(actual_lick_vector)

    # actual_lick = actual_lick_vector

    cross_entropy = -(np.dot(actual_lick_vector, np.log(p_lick)) + np.dot((1 - actual_lick_vector), np.log(1-p_lick)))
    return cross_entropy

VALID_VALUE = 0.88888888

def matrix_cross_entropy_loss(lick_matrix, prediction_matrix):
    mask = np.where(lick_matrix == 99, 0, 1)
    prediction_matrix_mask = np.where(prediction_matrix == 0.88888888, 0, 1)
    cross_entropy = - ((lick_matrix * np.log(prediction_matrix)) + (1 - lick_matrix) * np.log(1-prediction_matrix))
    cross_entropy = cross_entropy * mask * prediction_matrix_mask

    return np.nansum(cross_entropy)  # nansum is just a quick fix, likely need to be more principled...


def matrix_weighted_cross_entropy_loss(lick_matrix, prediction_matrix, alpha=1, cumulative_lick=False):
    mask = np.where(lick_matrix == 99, 0, 1)
    prediction_matrix_mask = np.where(prediction_matrix == 0.88888888, 0, 1)
    # cross_entropy = -(alpha * lick_matrix * np.log(prediction_matrix) + (1 - lick_matrix) * np.log(1-prediction_matrix))
    # cross_entropy = cross_entropy * mask * prediction_matrix_mask

    ################## Cross entropy on the cumulative p(lick) ###########################
    if cumulative_lick is True:
        small_value = 1e-5 # can't be zero because then the cross entropy will have to do log(0)
        prediction_matrix = np.where(prediction_matrix == 0.88888888, small_value, prediction_matrix) # time shift p(lick) = 0
        prediction_matrix = batched_cumulative_lick(None, prediction_matrix)
        prediction_matrix = np.where(prediction_matrix >=1, 1-small_value, prediction_matrix)
        prediction_matrix = np.where(prediction_matrix <=0, small_value, prediction_matrix)

    ################# Cross Entropy on the Instataneous p(lick) #######################
    # modifying cross entropy to prevent NaNs (probably due to underflow of prediction_matrix?)
    # small_value = 0 # originally 1e-9
    # cross_entropy = - (alpha * lick_matrix * np.log(np.maximum(prediction_matrix, small_value)) +
    #                   (1 - lick_matrix) * np.log(1 - np.maximum(prediction_matrix, small_value))
    #                   )

    # hard threshold prediction matrix
    small_value = 1e-5
    prediction_matrix = np.where(prediction_matrix >= 1, 1 - small_value, prediction_matrix)
    prediction_matrix = np.where(prediction_matrix <= 0, small_value, prediction_matrix)

    cross_entropy = - (alpha * lick_matrix * np.log(prediction_matrix) +
                       (1 - lick_matrix) * np.log(1 - prediction_matrix)
                       )

    cross_entropy = cross_entropy * mask * prediction_matrix_mask

    return np.sum(cross_entropy)


def loss_function_fit_vector(param_vals):
    # missing arguments: signals, param_vals_, actual_lick_,
    """
    loss between predicted licks and actual licks
    computed using cross entropy
    This loss function is for licking versus no-licking, but does not compute the loss of the reaction time.
    :param: param_vals, array of parameter values, in the following order:
        1. k parameter in the sigmoid function
        2. false_negative value in cost-benefit function
        3. midpoint of the sigmoid decision function
    :return:
    """

    cross_entropy_loss = 0
    for x_num, x in enumerate(signals):
        posterior = forward_inference(x.flatten())

        p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
                                    false_negative=param_vals[1], false_positive=0.0)
        p_lick = apply_strategy(p_lick, k=param_vals[0], midpoint=param_vals[2])
        # p_lick = np.max(p_lick) # this is the main difference between this function and loss_functinon

        # apply time shift to the posterior, relies time_shift global var
        # pad on the left (make model slower)

        # p_lick = np.pad(p_lick, (time_shift, 0), "constant", constant_values=(0, ))
        # padding doesn't work with some versions of autograd
        # p_lick = np.concatenate([np.zeros(time_shift), p_lick])
        baseline = 0.01
        p_lick = np.concatenate([np.repeat(baseline, time_shift), p_lick])
        p_lick = p_lick[0:len(actual_lick_vector[x_num])]  # clip p(lick) to be same length as actual lick


        # Caution: if k is quite lager, then the sigmoid may output 0, and np.log() will return an error.

        # TODO: pad on the right (make model faster) (actually just take window)
        cross_entropy_loss += np.sum(-(actual_lick_vector[x_num] * np.log(p_lick) + (1 - actual_lick_vector[x_num]) * np.log(
            1 - p_lick)))  # sum over the entire time series

        # print(x_num, cross_entropy_loss)

        # TODO: likely have to be weighted cross-entropy

    return cross_entropy_loss

def run_through_dataset_fit_vector(datapath, savepath, training_savepath, param=None, num_non_hazard_rate_param=2,
                                   fit_hazard_rate=True,
                                   cv=False,
                                   epoch_index=None, t_shift=0):
    """
    Takes in the experimental data, and makes inference of p(lick) for each time point given the stimuli
    :param datapath:
    :param savepath:
    :param param:
    :return:
    """

    global time_shift
    time_shift = 6

    if param is None:
        # if no parameters specified, then load the training result and get the last param
        with open(training_savepath, "rb") as handle:
            training_result = pkl.load(handle)

    if epoch_index is None and cv is False:
        min_loss_epoch_index = onp.where(training_result["loss"] == min(training_result["loss"]))[0][0]
        epoch_index = min_loss_epoch_index
    elif epoch_index is None and cv is True and time_shift is None:
        min_val_loss_epoch_index = onp.where(training_result["val_loss"] == min(training_result["val_loss"]))[0][0]
        epoch_index = min_val_loss_epoch_index
    elif epoch_index is None and cv is True and time_shift is not None:
        # get minimum validation loss for a particular time shift
        time_shift_index = onp.where(onp.array(training_result["time_shift"]) == time_shift)[0]
        min_val_loss = min(training_result["mean_val_loss"][time_shift_index])
        epoch_index = onp.where(training_result["mean_val_loss"] == min_val_loss)[0][0]

    # normally -1 (last training epoch)
    if cv is False:
        param = training_result["param_val"][epoch_index]
        print("Parameter training loss: ",  str(training_result["loss"][epoch_index]))  # just to double check
    else:
        param = training_result["param_val"][epoch_index]
        print("Parameter training loss: ", str(training_result["val_loss"][epoch_index]))
        print("Mean validation loss:", str(training_result["mean_val_loss"][epoch_index]))

    # note that parameters do not need to be pre-processed
    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"].flatten()
    # mouse_rt = exp_data["rt"].flatten()
    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()
    absolute_decision_time = exp_data["rt"].flatten()
    # peri_stimulus_rt = absolute_decision_time - tau

    # time-varying hazard rate
    # _, transition_matrix_list = get_hazard_rate(hazard_rate_type="subjective", datapath=datapath)

    # Find out when the trial actually ends, and create a mask of that:
    # (1 means trial still happening, 0 means those are padded signals)
    # (onp is used here to allow assignment into array, which is valid because this procedure is not in gradient descent
    trial_duration_list = list()
    for s in signals:
        trial_duration_list.append(len(s[0]))

    num_trial = len(signals)
    max_duration = onp.max(trial_duration_list)

    trial_duration = onp.array(trial_duration_list) - 1 # convert to 0 indexing
    trial_duration_mask = onp.zeros(shape=(num_trial, max_duration))

    for n, trial_duration in enumerate(trial_duration_list):
        trial_duration_mask[n, :trial_duration] = 1

    signal_matrix, lick_matrix = create_vectorised_data(datapath)
    # batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    global num_non_hazard_rate_params
    num_non_hazard_rate_params = num_non_hazard_rate_param # this is for predict_lick_w_hazard_rate

    if fit_hazard_rate is True:
        batched_predict_lick = vmap(predict_lick_w_hazard_rate, in_axes=(None, 0))
        # For use when training a constant hazard rate
        global max_signal_length
        max_signal_length = np.shape(lick_matrix)[1]
    else:
        global transition_matrix_list
        # _, transition_matrix_list = get_hazard_rate(hazard_rate_type="constant", constant_val=0.0001,
        # datapath=datapath)
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental_instantaneous", datapath=datapath)
        batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    # signals has to be specially prepared to get the vectorised code running
    prediction_matrix = batched_predict_lick(param, signal_matrix)
    # prediction_matrix[lick_matrix == 99] = onp.nan

    # Prediction matrix truncated by the lick time of the mouse
    prediction_matrix = np.where(lick_matrix == 99, onp.nan, prediction_matrix)

    batched_cumulative_lick = vmap(predict_cumulative_lick, in_axes=(None, 0))
    if time_shift == 0:
        # if there is time shift, then can't use the initial padding values
        # batched_cumulative_lick = vmap(predict_cumulative_lick, in_axes=(0, None))
        cumulative_lick_matrix = batched_cumulative_lick(None, prediction_matrix)

    # full signal matrix
    full_signal_prediction_matrix = batched_predict_lick(param, signal_matrix)
    full_signal_prediction_matrix = np.where(trial_duration_mask == 0, onp.nan, full_signal_prediction_matrix)

    if time_shift != 0:
        # convert baseline to something sensible
        baseline_p_lick = 0.0001
        prediction_matrix = np.where(prediction_matrix == 0.88888888, baseline_p_lick, prediction_matrix)
        full_signal_prediction_matrix = np.where(full_signal_prediction_matrix == 0.88888888, baseline_p_lick,
                                                 full_signal_prediction_matrix)

        cumulative_lick_matrix = batched_cumulative_lick(None, prediction_matrix)


    vec_dict = dict()
    vec_dict["change_value"] = change
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = prediction_matrix
    vec_dict["model_vec_output_cumulative"] = cumulative_lick_matrix
    vec_dict["full_signal_model_vec_output"] = full_signal_prediction_matrix
    vec_dict["true_change_time"] = tau
    vec_dict["epoch_index"] = epoch_index

    savepath = savepath + "_time_shift_" + str(t_shift) + ".pkl"
    with open(savepath, "wb") as handle:
        pkl.dump(vec_dict, handle)


def control_model(datapath, savepath, training_savepath, cv_random_seed=None, hazard_rate=0.0001,
                  time_shift=None, fitted_params=None):
    """
    Obtain the instantaneous lick probability with the assumption that the mouse act directly from the posterior:
    p(lick) = p(z_change at k | x_1:k). There are no tunable parameters in this model.
    Hazard rate can be constant, or obtained from experimental data (but is not fitted).
    :param datapath:
    :param savepath: where the model output will be saved (ie. the p(lick) for use in sampling)
    :param training_savepath: where the model training result will be saved (training, validation, test loss)
    Since there are no fitted parameters in the control model, this will just be the loss of the model in one epoch.
    :param cv_random_seed : random seed for setting cross validation set
    :param hazard_rate: if hazard rate is a float, then it will be used a constant hazard rate value
    if it is a string - "experimental" - then uses the hazard rate obtained via trial data
    :param time_shift: time shift, if positive, then this is the delay between the posterior and p(lick)
    :return:
    """

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    # signals = exp_data["ys"].flatten()
    # mouse_rt = exp_data["rt"].flatten()
    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()
    absolute_decision_time = exp_data["rt"].flatten()

    # adding "subjective change value" based on whether the mouse actually saw the change period
    # (some trial ended early when the mouse licks)
    mouse_early_lick = onp.where(absolute_decision_time < tau)[0] # NaNs should not count, those are classify as misses
    subjective_change = onp.array(np.exp(exp_data["sig"].flatten()))
    subjective_change[mouse_early_lick] = 1

    # runs data through a control which uses the normative posterior (no transfer functions)
    signal_matrix, lick_matrix = create_vectorised_data(datapath)

    global transition_matrix_list
    if hazard_rate == "experimental":
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath)
    else:
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="constant", datapath=datapath,
                                                    constant_val=hazard_rate)

    global batched_predict_lick
    batched_predict_lick = vmap(predict_lick_control, in_axes=(None, 0))
    prediction_matrix = batched_predict_lick(None, signal_matrix)

    full_prediction_matrix = prediction_matrix.copy() # include signals after the mouse licked

    # Remove predictions after the mouse licks, since there are no real signals after the mouse licked
    prediction_matrix = np.where(lick_matrix == 99, onp.nan, prediction_matrix)

    # Get the training, cross-validation and test loss from the control model.
    if cv_random_seed is None:
        cv_random_seed = onp.random.seed()

    mouse_reaction = exp_data["outcome"].flatten()

    le = LabelEncoder()
    mouse_reaction = le.fit_transform(mouse_reaction.tolist())

    mouse_reaction_df = pd.DataFrame({'trial': np.arange(0, len(mouse_reaction)),
                                      'outcome': mouse_reaction})
    y = mouse_reaction_df["outcome"]

    # y_train and y_test are just placeholders. This is only used to obtain the indices.
    X_dev, X_test, y_dev, y_test = train_test_split(mouse_reaction_df, y, test_size=0.1, random_state=cv_random_seed,
                                                    stratify=y)

    # further split validation set
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1,
                                                      random_state=cv_random_seed, stratify=y_dev)

    signal_matrix_test = signal_matrix[X_test["trial"], :]
    signal_matrix_val = signal_matrix[X_val["trial"], :]
    signal_matrix_train = signal_matrix[X_train["trial"], :]

    lick_matrix_test = lick_matrix[X_test["trial"], :]
    lick_matrix_val = lick_matrix[X_val["trial"], :]
    lick_matrix_train = lick_matrix[X_train["trial"], :]


    training_result = dict()
    set_names = ["test", "val", "train"]
    for s_matrix, l_matrix, set_name in zip([signal_matrix_test, signal_matrix_val, signal_matrix_train],
                                          [lick_matrix_test, lick_matrix_val, lick_matrix_train],
                                            set_names):
        model_output = batched_predict_lick(None, s_matrix)
        model_output = np.where(l_matrix == 99, onp.nan, model_output)  # remove model output with no mouse data
        model_loss = matrix_weighted_cross_entropy_loss(l_matrix, model_output, alpha=1, cumulative_lick=False)
        training_result[set_name + "_loss"] = model_loss



    training_result["param_val"] = hazard_rate
    training_result["time_shift"] = "None"
    training_result["epoch"] = 1
    training_result["batch_size"] = "None"
    training_result["step_size"] = "None"
    # TODO: things here can be put into the loop
    training_result["train_set_size"] = len(y_train)
    training_result["val_set_size"] = len(y_val)
    training_result["test_set_size"] = len(y_test)
    training_result["mean_train_loss"] = training_result["train_loss"] / len(y_train)
    training_result["mean_val_loss"] = training_result["val_loss"] / len(y_val)
    training_result["mean_test_loss"] = training_result["test_loss"] / len(y_test)

    if fitted_params is None:
        training_result["fitted_params"] = "None"
    else:
        training_result["fitted_params"] = fitted_params

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)

    # Save model output

    vec_dict = dict()
    vec_dict["change_value"] = change
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = prediction_matrix
    vec_dict["full_signal_model_vec_output"] = full_prediction_matrix
    vec_dict["true_change_time"] = tau
    vec_dict["subjective_change"] = subjective_change
    vec_dict["hazard_rate"] = hazard_rate

    with open(savepath, "wb") as handle:
        pkl.dump(vec_dict, handle)


def plot_training_result(training_savepath):
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    loss = np.concatenate(training_result["loss"])
    param_val = np.concatenate(training_result["param_val"])

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(np.arange(1, len(loss) + 1), -loss)
    axs[1].plot(np.arange(1, len(loss) + 1), param_val)

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Parameter value")

    plt.show()


def compare_model_with_behaviour(model_behaviour_df_path, savepath=None, showfig=True):
    model_behaviour_df = pd.read_pickle(model_behaviour_df_path)

    # plot psychometric curve

    model_prop_choice = model_behaviour_df.groupby(["change"], as_index=False).agg({"decision": "mean"})
    mouse_prop_choice = model_behaviour_df.groupby(["change"], as_index=False).agg({"mouse_lick": "mean"})

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot model behaviour
    ax.plot(model_prop_choice.change, model_prop_choice.decision)
    ax.scatter(model_prop_choice.change, model_prop_choice.decision)

    # Plot mouse behaviour
    ax.plot(mouse_prop_choice.change, mouse_prop_choice.mouse_lick)
    ax.scatter(mouse_prop_choice.change, mouse_prop_choice.mouse_lick)

    ax.legend(["Model", "Mouse"], frameon=False)

    ax.set_ylim([-0.05, 1.05])

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel("P(lick)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()


def test_vectorised_inference(exp_data_path):

    global param_vals
    param_vals = [10, 0, 0.5]

    global signal_matrix
    global lick_matrix
    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path)
    print("Signal matrix shape:", onp.shape(signal_matrix))
    print("Lick matrix shape:", onp.shape(lick_matrix))

    print("Number of nans in signal matrix", onp.sum(onp.isnan(signal_matrix)))
    print("Number of nans in lick matrix:", onp.sum(onp.isnan(lick_matrix)))

    global batched_predict_lick
    batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))
    # batched_cross_entropy = vmap(cross_entropy_loss, in_axes=(None, 0))
    # (for some reason, vmap for predict_lick must take two arguments, one with the parameters and one with the signals)

    # predictions = batched_predict_lick(param_vals, signal_matrix)
    # print("Shape of predictions", np.shape(predictions))
    # loss = batched_cross_entropy(lick_matrix, predictions)

    # loss = matrix_cross_entropy_loss(lick_matrix, predictions)
    # print("Loss:", loss)
    # for l, p in zip(lick_matrix, predictions):
    #     loss = cross_entropy_loss(l, p)
    #     print(loss)

    loss = loss_function_fit_vector_faster(param_vals)
    print("Loss from loss function:", loss)


def test_loss_function(datapath, plot_example=True, savepath=None):

    params = [10.0, 0.5, 0.0]
    time_shift = 0

    signal_matrix, lick_matrix = create_vectorised_data(datapath)
    batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    # Plot examples
    if plot_example is True:
        for s, l in zip(signal_matrix, lick_matrix):
            prediction = predict_lick(params, s)
            loss = matrix_cross_entropy_loss(lick_matrix=l, prediction_matrix=prediction)
            biased_loss = matrix_weighted_cross_entropy_loss(lick_matrix=l, prediction_matrix=prediction, alpha=20)

            fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            axs[0].plot(s[l != 99])
            axs[1].plot(l[l != 99])
            axs[1].plot(prediction[l != 99])
            axs[1].text(x=0.1, y=0.8, s="Loss: " + str(loss), fontsize=12)
            axs[1].text(x=0.1, y=0.6, s="Biased loss: " + str(biased_loss), fontsize=12)
            axs[1].legend(["Mouse lick", "Model prediction"], frameon=False)

            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[1].spines["top"].set_visible(False)
            axs[1].spines["right"].set_visible(False)

            axs[0].set_ylabel("Stimulus speed")
            axs[1].set_xlabel("Time (frames)")
            axs[1].set_ylabel("P(lick)")

            axs[1].set_ylim([0, 1])


            plt.show()

    # Plot loss across all examples

    all_zero_prediction_matrix = onp.zeros(shape=np.shape(lick_matrix))
    all_zero_prediction_matrix[:] = 1e-8

    all_one_prediction_matrix = onp.zeros(shape=np.shape(lick_matrix))
    all_one_prediction_matrix[:] = 1 - 1e-8


    alpha_list = np.arange(1, 20)
    # num_model = 3
    # loss_store = np.zeros(shape=(num_model, len(alpha_list)))
    all_zero_loss_store = list()
    all_one_loss_store = list()
    model_loss_store = list()

    prediction_matrix = batched_predict_lick(params, signal_matrix)

    for n, alpha in tqdm(enumerate(alpha_list)):
        all_zero_loss = matrix_weighted_cross_entropy_loss(lick_matrix, all_zero_prediction_matrix, alpha=alpha)
        # all_one_loss = matrix_weighted_cross_entropy_loss(lick_matrix, all_one_prediction_matrix, alpha=alpha)
        model_loss = matrix_weighted_cross_entropy_loss(lick_matrix, prediction_matrix, alpha=alpha)

        all_zero_loss_store.append(all_zero_loss)
        # all_one_loss_store.append(all_one_loss)
        model_loss_store.append(model_loss)
        # loss_store[0, n] = all_zero_loss
        # loss_store[1, n] = all_one_loss
        # loss_store[2, n] = model_loss

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    # axs.plot(loss_store)
    axs.plot(alpha_list, all_zero_loss_store)
    # axs.plot(alpha_list, all_one_loss_store)
    axs.plot(alpha_list, model_loss_store)
    axs.legend(["All zero", "Model"], frameon=False)

    axs.set_ylabel("Cross entropy loss")
    axs.set_xlabel("False negative bias (alpha)")

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def get_hazard_rate(hazard_rate_type="subjective", datapath=None, plot_hazard_rate=False, figsavepath=None,
                    constant_val=0.001):
    """
    The way I see it, there are 3 types of varying hazard rate.
    1. "normative": One specified by the experimental design;
    the actual distribution where the trial change-times are sampled from
    2. "experimental": The distribution in the trial change-times
    3. "subjective": The distribution experienced by the mice
    :param hazard_rate_type:
    :param datapath:
    :param plot_hazard_rate:
    :param figsavepath:
    :return:
    """


    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    change_time = exp_data["change"].flatten()
    signal = exp_data["ys"].flatten()
    num_trial = len(change_time)
    signal_length_list = list()
    for s in signal:
        signal_length_list.append(len(s[0]))

    max_signal_length = max(signal_length_list)

    # the following experimental hazard rate calculation sums to 1 (hist)
    # experimental_hazard_rate = onp.histogram(change_time, range=(0, max_signal_length), bins=max_signal_length)[0]
    # experimental_hazard_rate = experimental_hazard_rate / num_trial

    # experimental hazard rate as defined in Janssen and Shadlen (2005)
    hazard_rate_hist = onp.histogram(change_time, range=(0, max_signal_length), bins=max_signal_length)[0] \
                       / float(num_trial)
    hazard_rate_cumsum = onp.cumsum(hazard_rate_hist, dtype="float64")
    hazard_rate_cumsum = onp.where(onp.array(hazard_rate_cumsum) > 1, 1, hazard_rate_cumsum)  # cumsum precision
    experimental_hazard_rate = hazard_rate_hist / (1.0 - hazard_rate_cumsum)
    experimental_hazard_rate = onp.where(~onp.isfinite(experimental_hazard_rate), 1,
                                         experimental_hazard_rate)  # remove divide by zero Inf/NaN
    experimental_hazard_rate = onp.where(experimental_hazard_rate > 1, 1,
                                         experimental_hazard_rate)

    # log implementation of the hazard rate
    # log_experimenetal_hazard_rate = np.log(hazard_rate_hist) - np.log(1 - hazard_rate_cumsum)
    # experimental_hazard_rate = log_experimenetal_hazard_rate # np.exp(log_experimenetal_hazard_rate)


    if hazard_rate_type == "subjective":
        # get change times
        # remove change times where the mice did a FA, or a miss
        outcome = exp_data["outcome"].flatten()
        hit_index = onp.where(outcome == "Hit")[0]
        num_subjective_trial = len(hit_index)
        subjective_change_time = change_time[hit_index]
        subjective_hazard_rate = onp.histogram(subjective_change_time, range=(0, max_signal_length),
                                               bins=max_signal_length)[0]

        # convert from count to proportion (Divide by num_trial or num_subjective_trial?)
        subjective_hazard_rate = subjective_hazard_rate / num_subjective_trial

        assert len(subjective_hazard_rate) == max_signal_length

        hazard_rate_vec = subjective_hazard_rate

    elif hazard_rate_type == "experimental":
        hazard_rate_vec = experimental_hazard_rate
    elif hazard_rate_type == "normative":
        pass
    elif hazard_rate_type == "constant":
        hazard_rate_constant = constant_val
        hazard_rate_vec = np.repeat(hazard_rate_constant, max_signal_length)
    elif hazard_rate_type == "random":
        hazard_rate_vec = onp.random.normal(loc=0.0, scale=0.5, size=(max_signal_length, ))
        hazard_rate_vec = standard_sigmoid(hazard_rate_vec)
    elif hazard_rate_type == "experimental_instantaneous":
        hazard_rate_hist = onp.histogram(change_time, range=(0, max_signal_length), bins=max_signal_length)[0] \
                           / float(num_trial)
        hazard_rate_vec = hazard_rate_hist
    if plot_hazard_rate is True:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(subjective_hazard_rate, label="Subjective hazard rate")
        # ax.plot(experimental_hazard_rate, label="Experiment hazard rate")
        ax.plot(hazard_rate_vec)

        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("P(change)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if figsavepath is not None:
            plt.savefig(figsavepath, dpi=300)
        plt.show()

    # Make the transition matrix based on hazard_rate_vec
    transition_matrix_list = list()
    for hazard_rate in hazard_rate_vec:
        transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0,
                                       hazard_rate/5.0, hazard_rate/5.0],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])
        transition_matrix_list.append(transition_matrix)

    return hazard_rate_vec, transition_matrix_list


def get_hazard_rate_new(exp_data, hazard_rate_type="subjective", plot_hazard_rate=False, figsavepath=None,
                    constant_val=0.001):
    """
    The way I see it, there is 3 types of varying hazard rate.
    1. "normative": One specified by the experimental design;
    the actual distribution where the trial change-times are sampled from
    2. "experimental": The distribution in the trial change-times
    3. "subjective": The distribution experienced by the mice
    :param hazard_rate_type:
    :param datapath:
    :param plot_hazard_rate:
    :param figsavepath:
    :return:
    """

    change_time = exp_data["change"].flatten()
    signal = exp_data["ys"].flatten()
    num_trial = len(change_time)
    signal_length_list = list()
    for s in signal:
        signal_length_list.append(len(s[0]))

    max_signal_length = max(signal_length_list)

    experimental_hazard_rate = onp.histogram(change_time, range=(0, max_signal_length), bins=max_signal_length)[0]
    experimental_hazard_rate = experimental_hazard_rate / num_trial

    if hazard_rate_type == "subjective":
        # get change times
        # remove change times where the mice did a FA, or a miss
        outcome = exp_data["outcome"].flatten()
        hit_index = onp.where(outcome == "Hit")[0]
        num_subjective_trial = len(hit_index)
        subjective_change_time = change_time[hit_index]
        subjective_hazard_rate = onp.histogram(subjective_change_time, range=(0, max_signal_length), bins=max_signal_length)[0]

        # convert from count to proportion (Divide by num_trial or num_subjective_trial?)
        subjective_hazard_rate = subjective_hazard_rate / num_subjective_trial

        assert len(subjective_hazard_rate) == max_signal_length

        hazard_rate_vec = subjective_hazard_rate

    elif hazard_rate_type == "experimental":
        hazard_rate_vec = experimental_hazard_rate
    elif hazard_rate_type == "normative":
        pass
    elif hazard_rate_type == "constant":
        hazard_rate_constant = constant_val
        hazard_rate_vec = np.repeat(hazard_rate_constant, max_signal_length)
    elif hazard_rate_type == "random":
        hazard_rate_vec = onp.random.normal(loc=0.0, scale=0.5, size=(max_signal_length, ))
        hazard_rate_vec = standard_sigmoid(hazard_rate_vec)
    if plot_hazard_rate is True:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(subjective_hazard_rate, label="Subjective hazard rate")
        ax.plot(experimental_hazard_rate, label="Experiment hazard rate")

        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("P(change)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if figsavepath is not None:
            plt.savefig(figsavepath, dpi=300)
        plt.show()

    # Make the transition matrix based on hazard_rate_vec
    transition_matrix_list = list()
    for hazard_rate in hazard_rate_vec:
        transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])
        transition_matrix_list.append(transition_matrix)

    return hazard_rate_vec, transition_matrix_list


def make_transition_matrix(hazard_rate_vec, backward_prob_vec):
    """
    Make a list of transition matrices for each time point, determined by the length of hazard_rate_vec
    :param hazard_rate_vec:
    :param backward_prob:
    :return:
    """

    # standard logistic function to contrain values to [0, 1]
    # hazard_rate_vec = standard_sigmoid(hazard_rate_vec)

    # Custom logistic function to constrain values
    small_value = 0.01
    hazard_rate_vec = nonstandard_sigmoid(hazard_rate_vec, min_val=0, max_val=1, k=1, midpoint=2)
    backward_prob_vec = nonstandard_sigmoid(backward_prob_vec, min_val=0, max_val=1, k=1, midpoint=2)

    # softmax
    # hazard_rate_vec = softmax(hazard_rate_vec)

    # Directly clip the values
    # small_value = 1e-12
    # hazard_rate_vec = np.where(hazard_rate_vec <= 0, small_value, hazard_rate_vec)
    # hazard_rate_vec = np.where(hazard_rate_vec >= 1, 1-small_value, hazard_rate_vec)

    transition_matrix_list = list()
    for hazard_rate, b in zip(hazard_rate_vec, backward_prob_vec):
        transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0,
                                       hazard_rate/5.0, hazard_rate/5.0],
                                  [b, 1 - b, 0, 0, 0, 0],
                                  [b, 0, 1 - b, 0, 0, 0],
                                  [b, 0, 0, 1 - b, 0, 0],
                                  [b, 0, 0, 0, 1 - b, 0],
                                  [b, 0, 0, 0, 0, 1 - b]])
        transition_matrix_list.append(transition_matrix)

    return transition_matrix_list

#################################### FITTING HAZARD RATE ETC. #########################################################

def gradient_descent_w_hazard_rate(exp_data_path, training_savepath, init_param_vals=np.array([10.0, 0.5]),
                                time_shift_list=np.arange(0, 5), num_epoch=10, fit_hazard_rate=True,
                                   n_params=2, batch_size=None):

    global num_non_hazard_rate_params
    num_non_hazard_rate_params = n_params



    # Define global variables used by loss_function
    global signal_matrix
    global lick_matrix

    # signal_matrix, lick_matrix = create_vectorised_data(exp_data_path)
    # Lick now has exponential decay.
    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path,
                                                        lick_exponential_decay=True, decay_constant=1.0)
    # initialise hazard rate using experimental values
    hazard_rate, _ = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_path)

    # add a small vlaue so that none of them are 0 (prevent early clipping)
    # small_value = 0.01
    # hazard_rate = hazard_rate + small_value

    # initialise hazard rate randomly
    # hazard_rate, _ = get_hazard_rate(hazard_rate_type="random", datapath=exp_data_path)
    
    # random initialisation
    # key = random.PRNGKey(777)
    # hazard_rate_random = random.normal(key, shape=(len(hazard_rate), ))
    # small_baseline = 1e-5
    # hazard_rate = np.where(hazard_rate<=0, small_baseline, hazard_rate)

    if fit_hazard_rate is True:
        init_param_vals = np.concatenate([init_param_vals, hazard_rate])

    global time_shift

    # loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))
    param_list = list() # list of list, each list is the parameter values for a particular time shift
    loss_val_list = list()
    # define batched prediction function using vmap
    global batched_predict_lick

    if fit_hazard_rate is True:
        batched_predict_lick = vmap(predict_lick_w_hazard_rate, in_axes=(None, 0))
    else:
        batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))
        global transition_matrix_list
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="constant", datapath=exp_data_path,
                                                    constant_val=0.0001)

    # for cases where the cumulative lick is used
    global batched_cumulative_lick
    batched_cumulative_lick = vmap(predict_cumulative_lick, in_axes=(None, 0))

    # print("Initial parameters:", init_param_vals)

    # config.update("jax_debug_nans", True) # nan debugging
    # COMMENT OUT UNLESS DEBUGGING; it causes slowdowns.

    # float 64 numbers to increase precision (perhaps will prevent underflow???)
    # config.update("jax_enable_x64", True)

    for time_shift in tqdm(time_shift_list):
        print("Time shift: ", str(time_shift))

        step_size = 0.01 # orignally 0.01
        momentum_mass = 0.4
        # opt_init, opt_update = optimizers.momentum(step_size, mass=momentum_mass)
        opt_init, opt_update = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-8)

        """
        @jit
        def update(i, opt_state):
            params = optimizers.get_params(opt_state)
            return opt_update(i, grad(loss_function_fit_vector_faster)(params), opt_state)
        """

        itercount = itertools.count()
        """
        for epoch in range(num_epoch):
            opt_state = update(next(itercount), opt_state)
            params = optimizers.get_params(opt_state)
            loss_val = loss_function_fit_vector_faster(params)
            print("Loss:", loss_val)
            print("Parameters:", params)
            param_list.append(params)
            loss_val_list.append(loss_val)
        """

        # Test loss function
        # loss_val = loss_function_batch(init_param_vals, (signal_matrix, lick_matrix))
        # print("Loss:", loss_val)

        # TODO: Add train-val-test set split


        # Minibatch gradient descent
        if batch_size is not None:
            num_train = onp.shape(signal_matrix)[0]
            num_complete_batches, leftover = divmod(num_train, batch_size)
            num_batches = num_complete_batches + bool(leftover)

            def data_stream():
                rng = npr.RandomState(0)
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                        yield signal_matrix[batch_idx, :], lick_matrix[batch_idx, :]
            batches = data_stream()


        opt_state = opt_init(init_param_vals)

        @jit
        def step(i, opt_state, batch):
            params = optimizers.get_params(opt_state)
            # print("Params within step:", params)
            # g = grad(loss_function_fit_vector_faster)(params)
            g = grad(loss_function_batch)(params, batch)
            # print(np.sum(g))
            return opt_update(i, g, opt_state)

        for epoch in range(num_epoch):
            # print("Epoch: ", str(epoch))
            if batch_size is not None:
                for _ in range(num_batches):
                    opt_state = step(epoch, opt_state, next(batches))
            else:
                opt_state = step(epoch, opt_state, (signal_matrix, lick_matrix))

            if epoch % 10 == 0:
                params = optimizers.get_params(opt_state)
                # print("Parameters", params)
                # loss_val = loss_function_fit_vector_faster(params)
                loss_val = loss_function_batch(params, (signal_matrix, lick_matrix))
                print("Loss:", loss_val)
                loss_val_list.append(loss_val)
                param_list.append(params)
                # print("Parameters:", params)

                # TODO: Write function to auto-stop on convergence

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = param_list
    # TODO: Also save the time shift number and epoch number


    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def gradient_descent_w_cv(exp_data_path, training_savepath, init_param_vals=np.array([10.0, 0.5]),
                                   time_shift_list=np.arange(0, 5), num_epoch=10, fit_hazard_rate=True,
                                   cv_random_seed=None,
                                   n_params=2, batch_size=None,
                                   fitted_params=None, patience_threshold=2, smoothing_lambda=0):
    """
    Runs gradient descent to optimise parameters of the HMM to fit behavioural results with cross validation.
    :param exp_data_path: path to file containing the experimental data
    :param training_savepath: path to save the training data results
    :param init_param_vals: initial parameter values
    :param time_shift_list: list of time shift to loop over
    :param num_epoch: number of epochs to run gradient descent
    :param fit_hazard_rate: if True, fits hazard rate parameters
    :param cv_random_seed: random seed used for test-validation-train split
    :param n_params: number of non-hazard-rate parameters to fit, subsequent parameters are hazard rate parameters
    :param batch_size: size of batch for minibatch gradient descent, if None, then the entire batch is used
    :param smoothing_lmabda: penalty term magnitude for smoothing the hazard rate
    :param fitted_params: (optional) list containing the names of the listed parameters, useful for model comparison
    :return:
    """
    # TODO: remove the globals
    global num_non_hazard_rate_params
    num_non_hazard_rate_params = n_params

    # Define global variables used by loss_function
    global signal_matrix
    global lick_matrix

    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path,
                                                        lick_exponential_decay=False, decay_constant=1.0)
    # initialise hazard rate using experimental values
    hazard_rate, _ = get_hazard_rate(hazard_rate_type="experimental_instantaneous", datapath=exp_data_path)

    # fit constant hazard rate
    # hazard_rate = np.array([0.01])

    # add a bit of noise to it (so there are no identical values)
    # key = random.PRNGKey(777)
    # hazard_rate_random = random.normal(key, shape=(len(hazard_rate), )) * np.max(hazard_rate) * 0.1
    # hazard_rate = hazard_rate + np.abs(hazard_rate_random)


    if fit_hazard_rate is True:
        init_param_vals = np.concatenate([init_param_vals, hazard_rate])

    global time_shift
    # time_shift = 0  # For debugging only

    # loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))

    # define batched prediction function using vmap
    global batched_predict_lick

    if fit_hazard_rate is True:
        batched_predict_lick = vmap(predict_lick_w_hazard_rate, in_axes=(None, 0))
    else:
        batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))
        global transition_matrix_list
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental_instantaneous", datapath=exp_data_path)

    # for cases where the cumulative lick is used
    # global batched_cumulative_lick
    # batched_cumulative_lick = vmap(predict_cumulative_lick, in_axes=(None, 0))

    # prediction_matrix = batched_predict_lick(init_param_vals, signal_matrix)  # for debugging only

    # print("Initial parameters:", init_param_vals)

    # config.update("jax_debug_nans", True) # nan debugging
    # COMMENT OUT UNLESS DEBUGGING; it causes slowdowns.

    # float 64 numbers to increase precision (perhaps will prevent underflow???)
    # config.update("jax_enable_x64", True)

    with open(exp_data_path, "rb") as handle:
        exp_data = pkl.load(handle)

    if cv_random_seed is None:
        cv_random_seed = onp.random.seed()

    mouse_reaction = exp_data["outcome"].flatten()

    le = LabelEncoder()
    mouse_reaction = le.fit_transform(mouse_reaction.tolist())

    mouse_reaction_df = pd.DataFrame({'trial': np.arange(0, len(mouse_reaction)),
                                      'outcome': mouse_reaction})
    y = mouse_reaction_df["outcome"]

    data_indices = np.arange(0, np.shape(y)[0])

    # y_train and y_test are just placeholders. This is only used to obtain the indices.
    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(mouse_reaction_df, y, data_indices,
                                                                               test_size=0.1,
                                                                              random_state=cv_random_seed, stratify=y)

    # further split validation set
    X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(X_dev, y_dev, dev_indices,
                                                                                test_size=0.1,
                                                      random_state=cv_random_seed, stratify=y_dev)

    signal_matrix_test = signal_matrix[X_test["trial"], :]
    signal_matrix_val = signal_matrix[X_val["trial"], :]
    signal_matrix_train = signal_matrix[X_train["trial"], :]

    lick_matrix_test = lick_matrix[X_test["trial"], :]
    lick_matrix_val = lick_matrix[X_val["trial"], :]
    lick_matrix_train = lick_matrix[X_train["trial"], :]

    # TODO: get index of train, val, and test set so we know which trials were used
    # (This helps later plotting of fitting of test set)

    # For use when training a constant hazard rate
    global max_signal_length
    max_signal_length = np.shape(lick_matrix)[1]

    # Store model data
    param_list = list()  # list of list, each list is the parameter values for a particular time shift
    train_loss_list = list()
    val_loss_list = list()
    epoch_num_list = list()
    test_loss_list = list()
    time_shift_store = list()

    # Hyperparameters for training the model
    step_size = 0.02  # orignally 0.01
    opt_init, opt_update = optimizers.adam(step_size, b1=0.9, b2=0.999, eps=1e-8)

    for time_shift in tqdm(time_shift_list):
        print("Time shift: ", str(time_shift))

        # Initialise patience counter to track when to perform early stop
        patience_counter = 0

        # Minibatch gradient descent
        if batch_size is not None:
            num_train = onp.shape(signal_matrix_train)[0]
            num_complete_batches, leftover = divmod(num_train, batch_size)
            num_batches = num_complete_batches + bool(leftover)

            def data_stream():
                rng = npr.RandomState(0)
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                        yield signal_matrix_train[batch_idx, :], lick_matrix_train[batch_idx, :]

            batches = data_stream()

        opt_state = opt_init(init_param_vals)

        @jit
        def step(i, opt_state, batch):
            params = optimizers.get_params(opt_state)
            g = grad(loss_function_batch)(params, batch, smoothing_lambda)
            return opt_update(i, g, opt_state)

        for epoch in range(num_epoch):
            # print("Epoch: ", str(epoch))
            if batch_size is not None:
                for _ in range(num_batches):
                    opt_state = step(epoch, opt_state, next(batches))
                    # print("Parameters:", opt_state[0][0:6])
            else:
                opt_state = step(epoch, opt_state, loss_function_batch(params,
                                                                       (signal_matrix_train, lick_matrix_train)))

            if epoch % 10 == 0:
                params = optimizers.get_params(opt_state)
                train_loss = loss_function_batch(params, (signal_matrix_train, lick_matrix_train), smoothing_lambda)
                val_loss = loss_function_batch(params, (signal_matrix_val, lick_matrix_val), smoothing_lambda)
                print("Training loss:", train_loss)
                train_loss_list.append(train_loss)

                print("Validation loss:", val_loss)
                val_loss_list.append(val_loss)

                param_list.append(params)
                epoch_num_list.append(epoch)

                time_shift_store.append(time_shift)

                # Early stopping based on validation loss
                if len(val_loss_list) >= 2:
                    stop = early_stop(old_loss=val_loss_list[-2], new_loss=val_loss_list[-1],
                                      alpha=0.001)
                    patience_counter += stop
                if patience_counter >= patience_threshold:
                    test_loss = loss_function_batch(params, (signal_matrix_test, lick_matrix_test), smoothing_lambda)
                    print("Test loss: " + str(test_loss))
                    test_loss_list.append(test_loss)
                    break

            # get test loss at the end of training (note the 0-indexing)
            if epoch == (num_epoch-1):
                # TODO: use the best params... (instead of the last)
                test_loss = loss_function_batch(params, (signal_matrix_test, lick_matrix_test))
                print("Test loss: " + str(test_loss))
                test_loss_list.append(test_loss)

    training_result = dict()
    training_result["train_loss"] = train_loss_list
    training_result["val_loss"] = val_loss_list
    training_result["test_loss"] = test_loss_list
    training_result["param_val"] = param_list
    training_result["init_param_val"] = init_param_vals
    training_result["time_shift"] = time_shift_store
    training_result["epoch"] = epoch_num_list
    training_result["batch_size"] = batch_size
    training_result["step_size"] = step_size
    training_result["train_set_size"] = len(y_train)
    training_result["val_set_size"] = len(y_val)
    training_result["test_set_size"] = len(y_test)

    training_result["mean_train_loss"] = np.array(train_loss_list) / len(y_train)
    training_result["mean_val_loss"] = np.array(val_loss_list) / len(y_val)
    training_result["mean_test_loss"] = np.array(test_loss_list) / len(y_test)

    training_result["val_indices"] = val_indices
    training_result["train_indices"] = train_indices
    training_result["test_indices"] = test_indices
    training_result["cv_random_seed"] = cv_random_seed
    training_result["smoothing_lambda"] = smoothing_lambda

    if fitted_params is not None:
        training_result["fitted_params"] = fitted_params

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def standard_sigmoid(input):
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


def nonstandard_sigmoid(input, min_val=0.0, max_val=1.0, k=1, midpoint=0.5):
    # naive implementation
    # output = (max_val - min_val) / (1.0 + np.exp(-input)) + min_val

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                      (max_val - min_val) / (1.0 + np.exp(-(input - midpoint) * k)) + min_val,
                      (max_val - min_val) * np.exp((input - midpoint) * k) / (1.0 + np.exp((input - midpoint) * k))
                      + min_val)

    return output

def softmax(input_vec):
    # native implementation
    output = np.exp(input_vec) / np.sum(np.exp(input_vec))
    return output


def inverse_nonstandard_sigmoid(input, min_val=0.0, max_val=1.0):

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                      (max_val - min_val) / (1.0 + np.exp(-input)) + min_val,
                      (max_val - min_val) * np.exp(input) / (1.0 + np.exp(input)) + min_val)

    return output


def predict_lick_w_hazard_rate(param_vals, signal):
    # NOTE: This function also depends on the global parameter "time_shift"

    # impose some hard boundaries

    # fully tunable hazard rate (and optional tunable constant backward probability)
    # posterior = forward_inference_w_tricks(signal.flatten())
    global transition_matrix_list
    hazard_rate_params = param_vals[num_non_hazard_rate_params:]
    # backward_prob_vec = np.repeat(param_vals[2], np.shape(hazard_rate_params)[0])
    backward_prob_vec = np.repeat(0.0, np.shape(hazard_rate_params)[0])  # use 0 by default
    transition_matrix_list = make_transition_matrix(hazard_rate_vec=hazard_rate_params,
                                                    backward_prob_vec=backward_prob_vec)

    # ^ note this works due to zero-indexing. (if num=2, then we start from index 2, which is the 3rd param)

    # constant tunable hazard rate
    # hazard_rate_param = np.repeat(param_vals[num_non_hazard_rate_params:], max_signal_length)
    # backward_prob_vec = np.repeat(1, np.shape(hazard_rate_param)[0])
    # transition_matrix_list = make_transition_matrix(hazard_rate_vec=hazard_rate_param,
    # backward_prob_vec=backward_prob_vec)

    # add noise to the signal
    # key = random.PRNGKey(777)
    # signal = signal.flatten() + (random.normal(key, (len(signal.flatten()), )) * nonstandard_sigmoid(param_vals[2],
    #                                                                                   min_val=0.0, max_val=1.0))
    # multiply standard normal by constant: aX + b = N(au + b, a^2\sigma^2)
    posterior = forward_inference_custom_transition_matrix(signal)

    small_value = 1e-6
    posterior = np.where(posterior >= 1.0, 1-small_value, posterior) # I've seen
    # rounding errors where the output is 1.0000001 (strange)
    # Flagging just in case it as cosequences to other models...
    posterior = np.where(posterior <= 0.0, small_value, posterior)  # prevent underflow

    # no noise in the signal
    # posterior = forward_inference_custom_transition_matrix(signal.flatten())

    # p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=param_vals[3],
    #                             false_negative=param_vals[4], false_positive=param_vals[5]) # param_vals[1]
    p_lick = apply_strategy(posterior, k=param_vals[0], midpoint=param_vals[1])

    # Use posterior and p_lick directly
    # p_lick = posterior

     #p_lick = apply_strategy(p_lick, k=nonstandard_sigmoid(param_vals[0], min_val=0.0, max_val=20),
    # midpoint=nonstandard_sigmoid(param_vals[1], min_val=-10, max_val=10),
    #                         max_val=1, min_val=0)

    # Add time shift
    if time_shift > 0:
        baseline = 0.88888888 # nan placeholder
        p_lick = np.concatenate([np.repeat(baseline, time_shift), p_lick])
        p_lick = p_lick[0:len(signal.flatten())]  # clip p(lick) to be same length as actual lick
    elif time_shift < 0:
        # backward shift
        baseline = 0.88888888  # nan placecholder
        p_lick = np.concatenate([p_lick, np.repeat(baseline, abs(time_shift))])
        p_lick = p_lick[abs(time_shift):(len(signal.flatten()) + abs(time_shift))]

    # hard threshold to make sure there are no invalid values due to overflow/underflow
    # small_value = 1e-9
    # p_lick = np.where(p_lick <=0.0, small_value, p_lick)
    # p_lick = np.where(p_lick >=1.0, 1-small_value, p_lick)

    return p_lick


def predict_cumulative_lick(param, p_lick_instant):
    p_lick_cumulative_list = list()
    p_no_lick_cumulative_list = list()
    p_no_lick_instant = 1 - p_lick_instant

    p_lick_cumulative_list.append(p_lick_instant[0])
    p_no_lick_cumulative_list.append(1 - p_lick_instant[0])

    for time_step in np.arange(1, np.shape(p_lick_instant)[0]):
        p_no_lick_cumulative_list.append(p_no_lick_cumulative_list[time_step-1] * (1 - p_lick_instant[time_step]))
        p_lick_cumulative = p_no_lick_cumulative_list[time_step-1] * p_lick_instant[time_step] + \
                            p_lick_cumulative_list[time_step-1]

        p_lick_cumulative_list.append(p_lick_cumulative)

    return np.array(p_lick_cumulative_list)



def predict_lick_control(param, signal):
    posterior = forward_inference_custom_transition_matrix(signal.flatten())
    p_lick = np.where(posterior > 1, 1, posterior) # I've seen rounding errors where the output is 1.0000001 (strange)
    # Flagging just in case it as cosequences to other models...
    return p_lick


def get_trained_hazard_rate(training_savepath, num_non_hazard_rate_param=2, epoch_num=1,
                            param_process_method="sigmoid"):
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    trained_hazard_rate = training_result["param_val"][epoch_num][num_non_hazard_rate_param:]

    if param_process_method == "sigmoid":
        trained_hazard_rate = standard_sigmoid(trained_hazard_rate)

    return trained_hazard_rate


def benchmark_model(datapath, training_savepath, figsavepath=None, alpha=1):

    signal_matrix, lick_matrix = create_vectorised_data(datapath)
    # Dummy model
    all_zero_prediction_matrix = onp.zeros(shape=np.shape(lick_matrix))
    small_value = 0.01
    all_zero_prediction_matrix[:] = small_value

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    loss = training_result["loss"][-1]
    all_zero_loss = matrix_weighted_cross_entropy_loss(lick_matrix, all_zero_prediction_matrix, alpha=alpha)

    figure, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.bar(x=[1, 2], height=[all_zero_loss, loss], tick_label=["All " + str(small_value), "Trained model"])

    ax.set_ylabel("Cross entropy loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()





def gradient_clipping(gradients, type="L2_norm"):
    if type == "L2_norm":
        threshold = 10 # need to read what to set this to...
        new_gradients = gradients * threshold / (gradients ** 2 / len(gradients)) # TODO: should be l2 norm

    return new_gradients


def get_max_signal_length(exp_data):


    return max_signal_length


def get_model_posterior(exp_data, training_result, num_non_hazard_rate_params=2):

    min_val_loss = min(training_result["mean_val_loss"])
    epoch_index = onp.where(training_result["mean_val_loss"] == min_val_loss)[0][0]
    param_vals = training_result["param_val"][epoch_index]
    hazard_rate_params = param_vals[num_non_hazard_rate_params:]

    global transition_matrix_list
    #  note that make_transition_matrix internally passes the hazard_rate_params through a sigmoid
    transition_matrix_list = make_transition_matrix(hazard_rate_vec=hazard_rate_params,
                                                    backward_prob_vec=np.repeat(0.0, len(hazard_rate_params)))

    vmap_forward_inference = vmap(forward_inference_custom_transition_matrix)

    # posterior = list()

    signal_matrix, _ = create_vectorised_data_new(exp_data)

    posterior = vmap_forward_inference(signal_matrix)

    hazard_rate = nonstandard_sigmoid(hazard_rate_params, min_val=0, max_val=1, k=1, midpoint=2)

    return posterior, hazard_rate

def test_early_stop(validation_loss, alpha=0.001):

    stop_list = list()
    for epoch in np.arange(1, len(validation_loss)):
        stop = early_stop(validation_loss[epoch-1], validation_loss[epoch], alpha=alpha)
        stop_list.append(stop)

    return stop_list


def early_stop(old_loss, new_loss, alpha=0.0001):
    """
    Compares loss in the current versus previous epoch, and determines whether gradient descent can be terminated.
    By loss, this is usually the validation loss.
    :param old_loss:
    :param new_loss:
    :param alpha:
    :param patience: how many epochs of failing the criteria to stop gradient descent
    :return:
    """
    if new_loss > old_loss:
        stop = 1
    elif abs(old_loss - new_loss) < alpha * abs(old_loss):
        stop = 1
    else:
        stop = 0

    return stop

def test_rel_loss():
    rel_difference = list()
    for v_old, v_new in zip(val_loss[0:48], val_loss[1:49]):
        rel_difference.append(abs(v_old - v_new) / abs(v_old))

def get_train_val_test_set_indices(exp_data, random_seed):

    # open exp data (actual data needed due to stratification process)
    mouse_reaction = exp_data["outcome"].flatten()

    le = LabelEncoder()
    mouse_reaction = le.fit_transform(mouse_reaction.tolist())

    mouse_reaction_df = pd.DataFrame({'trial': np.arange(0, len(mouse_reaction)),
                                      'outcome': mouse_reaction})
    y = mouse_reaction_df["outcome"]

    data_indices = np.arange(0, np.shape(y)[0])

    # y_train and y_test are just placeholders. This is only used to obtain the indices.
    X_dev, X_test, y_dev, y_test, dev_set_indices, test_set_indices = train_test_split(
                                                          mouse_reaction_df, y, data_indices, test_size=0.1,
                                                          random_state=random_seed, stratify=y)

    # further split validation set
    X_train, X_val, y_train, y_val, train_set_indices, val_set_indices = train_test_split(
        X_dev, y_dev, dev_set_indices,_test_size=0.1, random_state=cv_random_seed, stratify=y_dev)


    return train_set_indices, val_set_indices, test_set_indices




def main(model_number=99, exp_data_number=83, run_test_foward_algorithm=False, run_test_on_data=False, run_gradient_descent=False,
         run_plot_training_loss=False, run_plot_sigmoid=False, run_plot_test_loss=False,
         run_model=False, run_plot_time_shift_test=False, run_plot_hazard_rate=False, run_plot_trained_hazard_rate=False,
         run_benchmark_model=False, run_plot_time_shift_training_result=False, run_plot_posterior=False, run_control_model=False,
         run_plot_signal=False, run_plot_trained_posterior=False, run_plot_trained_sigmoid=False, run_plot_time_shift_cost=False,
         run_plot_change_times=False, run_get_model_posterior=False, run_plot_early_stop=False, find_best_time_shift=False,
         blocktype=None, smoothing_lambda=None):
    home = expanduser("~")
    print("Running model: ", str(model_number))
    print("Using mouse: ", str(exp_data_number))

    # datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/subsetted_data/data_IO_083.pkl"
    main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    # TODO: generalise the code below
    if blocktype is None:
        datapath = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(exp_data_number) + ".pkl")
    elif blocktype == "early":
        datapath = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(exp_data_number) + "_early_blocks" ".pkl")
    elif blocktype == "late":
        datapath = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_0" + str(exp_data_number) + "_late_blocks" ".pkl")

    model_save_path = os.path.join(main_folder, "hmm_data/model_response_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".pkl")
    fig_save_path = os.path.join(main_folder, "figures/model_response_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".png")
    training_savepath = os.path.join(main_folder, "hmm_data/training_result_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".pkl")

    print("Saving training data to: ", training_savepath)
    print("Mouse data path: ", datapath)

    """
    param: 
    1. k parameter in the sigmoid function
    2. false_negative value in cost-benefit function
    3. midpoint of the sigmoid decision function
    """

    if run_test_foward_algorithm is True:
        test_forward_algorithm()

    if run_test_on_data is True:
        test_on_data(change_val=1.25, exp_data_file=datapath)

    if run_model is True:
        model_save_path = os.path.join(main_folder, "hmm_data/model_response_0" + str(exp_data_number) + "_"
                                       + str(model_number))
        run_through_dataset_fit_vector(datapath=datapath, savepath=model_save_path, training_savepath=training_savepath,
                                       num_non_hazard_rate_param=2, fit_hazard_rate=True,
                                       cv=True, param=None, t_shift=6)

    if run_control_model is True:
        control_model(datapath=datapath, savepath=model_save_path, training_savepath=training_savepath,
                      cv_random_seed=777)

    if run_gradient_descent is True:
        # gradient_descent_w_hazard_rate(exp_data_path=datapath,
        #                                      training_savepath=training_savepath,
        #                                      # init_param_vals=np.array([0.0, 0.1, 0.1]), # originally 10.0, 0.5, 0.1
        #                                      init_param_vals = np.array([10.0, 0.5]),
        #                                      n_params=2, fit_hazard_rate=True,
        #                                     time_shift_list=np.arange(1, 3), num_epoch=200, batch_size=128)
        # batch size originally 128

        # sim 32:  time_shift_list=np.arange(0, 22, 2), num_epoch=200

        gradient_descent_w_cv(exp_data_path=datapath,
                              training_savepath=training_savepath,
                              # init_param_vals = np.array([10.0, 0.5, -3, 0.2, 0.2, 0.2]),
                              init_param_vals=np.array([10.0, 0.5]),  # 0.2 (backward_prob)
                              n_params=2, fit_hazard_rate=True,
                              time_shift_list=np.arange(0, 11), num_epoch=500, batch_size=512,
                              cv_random_seed=777,
                              # fitted_params=["sigmoid_k", "sigmoid_midpoint", "stimulus_var",
                              #                "true_negative", "false_negative", "false_positive",
                              #                "hazard_rate", "backward_prob"]
                              fitted_params=["sigmoid_k", "sigmoid_midpoint", "hazard_rate",
                                            "time_shift"],
                              smoothing_lambda=smoothing_lambda,
                              )


    if run_plot_test_loss is True:
        figsavepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/test_alpha_model_"
                                   + str(model_number) + ".png")
        test_loss_function(datapath, plot_example=False, savepath=figsavepath)

    if run_plot_training_loss is True:
        figsavepath = os.path.join(main_folder, "figures/training_result_model" + str(model_number) + "_mouse_" +
                                   str(exp_data_number))
        nmt_plot.plot_training_loss(training_savepath, figsavepath=figsavepath, cv=True, time_shift=True)

    if run_plot_time_shift_cost is True:
        figsavepath = os.path.join(main_folder, "figures/time_shift_min_loss_model" + str(model_number))
        exp_data_number_list = [75, 78, 79, 80, 81, 83]
        label_list = [1, 2, 3, 4, 5, 6]
        plt.style.use(stylesheet_path)
        fig, ax = plt.subplots(figsize=(4, 4))
        for exp_data_number, label in zip(exp_data_number_list, label_list):
            training_savepath = os.path.join(main_folder, "hmm_data/training_result_0" + str(exp_data_number) + "_"
                                             + str(model_number) + ".pkl")
            with open(training_savepath, "rb") as handle:
                training_result = pkl.load(handle)

            fig, ax = nmt_plot.plot_time_shift_cost(fig, ax, training_result, label=label)

        ax.grid()
        ax.legend(title="Mouse")
        fig.savefig(figsavepath)

    if run_plot_sigmoid is True:
        figsavepath = os.path.join(main_folder, "figures/transfer_function_model" + str(model_number))
        nmt_plot.plot_sigmoid_comparisions(training_savepath, plot_least_loss_sigmoid=True, figsavepath=figsavepath)

    if run_plot_trained_sigmoid is True:
        exp_data_number_list = [75, 78, 79, 80, 81, 83]
        label_list = [1, 2, 3, 4, 5, 6]
        figsavepath = os.path.join(main_folder, "figures", "sigmoid",
                                   "mouse_sigmoid_comparison_model_" + str(model_number))
        plt.style.use(stylesheet_path)
        fig, ax = plt.subplots(figsize=(4, 4))
        sigmoid_func = functools.partial(apply_strategy, max_val=1.0, min_val=0.0, policy="sigmoid",
                                         epsilon=0.1, lick_bias=0.5)
        for exp_data_number, label in zip(exp_data_number_list, label_list):
            training_savepath = os.path.join(main_folder, "hmm_data/training_result_0" + str(exp_data_number) + "_"
                                             + str(model_number) + ".pkl")
            with open(training_savepath, "rb") as handle:
                training_result = pkl.load(handle)
            fig, ax = nmt_plot.plot_trained_sigmoid(fig, ax, training_result, sigmoid_func,
                                                    training_epoch=None, label=label)

        ax.grid()
        ax.legend(title="Mouse")
        fig.savefig(figsavepath)

        # plot trained sigmoid parameters
        figsavepath = os.path.join(main_folder, "figures", "sigmoid",
                                   "mouse_sigmoid_params_" + str(model_number) + "_beta")
        fig, ax = plt.subplots(figsize=(4, 4))
        for exp_data_number, label in zip(exp_data_number_list, label_list):
            training_savepath = os.path.join(main_folder, "hmm_data/training_result_0" + str(exp_data_number) + "_"
                                             + str(model_number) + ".pkl")
            with open(training_savepath, "rb") as handle:
                training_result = pkl.load(handle)
            fig, ax = nmt_plot.plot_trained_sigmoid_param(fig, ax, training_result, param_conversion_func=None,
                                                    training_epoch=None, label=label)

        ax.grid()
        ax.legend(title="Mouse")
        fig.savefig(figsavepath)

    if run_plot_trained_hazard_rate is True:
        figsavepath = os.path.join(main_folder, "figures/trained_hazard_rate_model_" + str(model_number) + "_mouse_"
                                   + str(exp_data_number))

        # get max_signal_length, for when constant hazard rate is used
        _, lick_matrix = create_vectorised_data(datapath)
        max_signal_length = np.shape(lick_matrix)[1]
        sigmoid_function = functools.partial(nonstandard_sigmoid, min_val=0, max_val=1.0, k=1, midpoint=0.5)
        fig, ax = nmt_plot.plot_trained_hazard_rate(training_savepath, sigmoid_function, num_non_hazard_rate_param=2,
                                                    constant_hazard_rate=False, max_signal_length=max_signal_length)
        fig.set_size_inches(8, 4)
        fig.savefig(figsavepath)

        # compare mouse and model
        figsavepath = os.path.join(main_folder, "figures/hazard_rate_comparison_model_" + str(model_number) + "_mouse_"
                                   + str(exp_data_number))
        fig, ax = nmt_plot.plot_trained_hazard_rate(training_savepath, sigmoid_function, num_non_hazard_rate_param=2,
                                                    constant_hazard_rate=False, max_signal_length=max_signal_length)
        mouse_hazard_rate, _ = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath,
                                               plot_hazard_rate=False,
                        figsavepath=None)
        ax.plot(mouse_hazard_rate)
        ax.legend(["Model", "Mouse"])
        fig.savefig(figsavepath)


    if run_plot_time_shift_test is True:
        for trial_num in np.arange(0, 5):
            nmt_plot.plot_time_shift_test(datapath, param=[10, 0.5], time_shift_list=[0, -10], trial_num=trial_num)

    if run_plot_hazard_rate is True:
        figsavepath = os.path.join(main_folder, "figures/hazard_rate_subjective_mouse_early_block" + str(exp_data_number) + ".png")
        # get_hazard_rate(hazard_rate_type="subjective", datapath=datapath, plot_hazard_rate=True,
        #                  figsavepath=figsavepath)

        mouse_hazard_rate, _ = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath,
                                               plot_hazard_rate=True,
                        figsavepath=figsavepath)




        # model_hazard_rate = get_trained_hazard_rate(training_savepath, num_non_hazard_rate_param=6,
        #                                             epoch_num=348, param_process_method="sigmoid")
        # model_hazard_rate = model_hazard_rate[6:-11]  # remove the time-shifted bits at the end
        # plt.plot(model_hazard_rate)
        # fig = nmt_plot.compare_model_mouse_hazard_rate(model_hazard_rate, mouse_hazard_rate,
        #                                                scale_method="sum")
        # plt.show()

    if run_benchmark_model is True:
        figsavepath = os.path.join(main_folder, "figures/loss_benchmark_model_" + str(model_number) + ".png")
        benchmark_model(datapath, training_savepath, figsavepath=figsavepath, alpha=1)

    if run_plot_time_shift_training_result is True:
        figsavepath = os.path.join(main_folder, "figures/time_shift_loss_model_" + str(model_number) + ".png")
        plot_time_shift_training_result(training_savepath=training_savepath, figsavepath=figsavepath,
                                        time_shift_list=np.arange(0, 10, 1),
                                        num_step=40, num_epoch_per_step=10
                                        )

    if run_plot_posterior is True:
        figsavepath = os.path.join(main_folder, "figures/pure_posterior_exp_transition_matrix_window_40.png")
        nmt_plot.plot_posterior(datapath=datapath, figsavepath=None)

    if run_plot_signal is True:
        nmt_plot.plot_signals(datapath)

    if run_plot_trained_posterior is True:
        figsavepath = os.path.join(main_folder, "figures/trained_posterior_realtime" + "_model_" +str(model_number) +
                                   "_mouse_" + str(exp_data_number))
        nmt_plot.plot_trained_posterior(datapath, training_savepath, plot_peri_stimulus=False,
                               num_examples=10, random_seed=777,
                               plot_cumulative=False,
                               figsavepath=figsavepath)

        figsavepath = os.path.join(main_folder, "figures/trained_posterior_peri_stimulus_time" + "_model_" +
                                   str(model_number) +
                                   "_mouse_" + str(exp_data_number))
        nmt_plot.plot_trained_posterior(datapath, training_savepath, plot_peri_stimulus=True,
                               num_examples=10, random_seed=777,
                               figsavepath=figsavepath, plot_cumulative=False)

    if run_plot_change_times is True:
        with open(datapath, "rb") as handle:
            exp_data = pkl.load(handle)

        figsavepath = os.path.join(main_folder, "figures/exp_data_plots/change_times" + "_mouse_" +
                                   str(exp_data_number))
        fig, ax = nmt_plot.plot_change_times(exp_data, xlim=[0, 350])
        fig.set_size_inches(4, 4)
        fig.savefig(figsavepath)

    if run_get_model_posterior is True:

        with open(training_savepath, "rb") as handle:
            training_result = pkl.load(handle)
        with open(datapath, "rb") as handle:
            exp_data = pkl.load(handle)

        posterior, hazard_rate = get_model_posterior(exp_data, training_result=training_result,
                                                     num_non_hazard_rate_params=2)

        model_posterior_save_path = os.path.join(main_folder, "hmm_data", "model_posterior_mouse_"
                                                 + str(exp_data_number) + "_model_" + str(model_number) + ".pkl")

        with open(model_posterior_save_path, "wb") as handle:
            pkl.dump(posterior, handle)

    if run_plot_early_stop is True:

        with open(training_savepath, "rb") as handle:
            training_result = pkl.load(handle)

        fig, ax = plt.subplots()
        num_time_shift = 11
        num_epoch = len(training_result["val_loss"]) / num_time_shift
        nmt_plot.plot_early_stopping(ax, validation_loss=onp.hstack(training_result["val_loss"][0:50]),
                                     stopping_criteria_func=early_stop)

    if find_best_time_shift is True:
        with open(training_savepath, "rb") as handle:
            training_result = pkl.load(handle)
        # TODO: Still needs to be tested.
        min_val_loss_index = onp.where(training_result["val_loss"] == min(training_result["val_loss"]))[0][0]
        best_time_shift = training_result["time_shift"][min_val_loss_index]
        print("Time shift with minimal validation loss:", str(best_time_shift))


if __name__ == "__main__":
    exp_data_number_list = [78, 79, 80, 81, 83]  # [75, 78, 79, 80, 81, 83]
    for exp_data_number in exp_data_number_list:
        main(model_number=80, exp_data_number=exp_data_number, run_test_on_data=False, run_gradient_descent=True,
             run_plot_training_loss=False, run_plot_sigmoid=False, run_plot_time_shift_cost=False,
             run_plot_test_loss=False, run_model=False, run_plot_time_shift_test=False,
             run_plot_hazard_rate=False, run_plot_trained_hazard_rate=False, run_benchmark_model=False,
             run_plot_time_shift_training_result=False, run_plot_posterior=False, run_control_model=False,
             run_plot_signal=False, run_plot_trained_posterior=False, run_plot_trained_sigmoid=False,
             run_plot_change_times=False, run_get_model_posterior=False, run_plot_early_stop=False,
             find_best_time_shift=False, blocktype=None, smoothing_lambda=1)

    # mouse_number = 75
    # smoothing_lambda_list = [0.1]
    # model_number_list = [81]
    # for model_number, smoothing_lambda in zip(model_number_list, smoothing_lambda_list):
    #     main(model_number=model_number, exp_data_number=mouse_number, run_gradient_descent=False,
    #          run_plot_training_loss=False, run_plot_trained_hazard_rate=False, find_best_time_shift=False,
    #          run_model=True,
    #          smoothing_lambda=smoothing_lambda)