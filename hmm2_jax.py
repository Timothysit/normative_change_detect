# Automatic Differentiation using jax
# https://github.com/google/jax
import jax.numpy as np
import numpy as onp # original numpy for indexed assignment/mutation (outside context of differentiation)
import numpy.random as npr # randomisation for minibatch gradient descent
from jax import grad, jit, vmap
from jax.experimental import optimizers
import jax.random as random

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
home = expanduser("~")
# home = "/home/timsit"

from scipy.signal import savgol_filter

# debugging nans returned by grad
from jax.config import config


def cal_p_x_given_z(x_k):
    z_mu = np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00]))
    z_var = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2 # in actual data, std is 0.25

    # z_mu = np.log(np.array([1.0, 1.25]))
    # z_var = np.array([0.25, 0.25])

    p_x_given_z = (1 / np.sqrt(2 * np.pi * z_var)) * np.exp(-(x_k - z_mu) ** 2 / (2 * z_var))

    # returns row vector (for simpler calculation later on)
    return p_x_given_z.T


def forward_inference(x):
    # rewritten to use without assignment of the form: A[i, j] = x
    # TODO: Think about logging thing to prevent underflow
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
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix_list[k-1])

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:])) # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    return np.array(p_change_given_x)


def apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=1.0, false_negative=1.0, false_positive=1.0):
    """
    computes decision value based on cost and benefit
    note that true-positive is always set to 1.0, everything else is relative to it
    :param true_positive:
    :return:
    """

    lick_benefit = change_posterior * true_positive
    lick_cost = (1 - change_posterior) * false_positive
    no_lick_benefit = (1 - change_posterior) * true_negative
    no_lick_cost = change_posterior * false_negative

    prob_lick = (lick_benefit + no_lick_benefit - no_lick_cost - lick_cost) / (true_positive +
                                                                               true_negative + false_negative + false_positive)

    return prob_lick


def apply_strategy(prob_lick, k=10, midpoint=0.5, max_val=1.0, min_val=0.0):
    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    # two param logistic function
    p_lick = (max_val - min_val) / (1 + np.exp(-k * (prob_lick - midpoint))) + min_val

    return p_lick


def plot_strategy(k_list):
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in k_list:
        prob_lick = np.linspace(0, 1, 1000)
        p_lick = apply_strategy(prob_lick, k)

        ax.plot(prob_lick, p_lick)
        ax.legend(k_list, frame=False)
        ax.set_xlabel("Input lick probability")
        ax.set_ylabel("Output lick probability")

    plt.show()


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


def plot_signal_and_inference(signal, tau, prob, savepath=None):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(signal)
    axs[0].axvline(tau, color="r", linestyle="--")

    axs[1].plot(prob)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

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

    # print("Signal baseline mean:", )

    # p_z_given_x = forward_inference(signal)


    # use custom transition matrix
    global transition_matrix_list
    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_file)
    p_z_given_x = forward_inference_custom_transition_matrix(signal)

    # Plot example
    plot_signal_and_inference(signal=signal, tau=tau, prob=p_z_given_x)

    return None


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


def create_vectorised_data(exp_data_path, subset_trials=None, lick_exponential_decay=False, decay_constant=0.5):
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
        signal_matrix[n, :len(s[0])] = s[0]
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
    signal = signal.flatten() + (random.normal(key, (len(signal.flatten()), )) * standard_sigmoid(param_vals[2]))
    # multiply standard normal by constant: aX + b = N(au + b, a^2\sigma^2)
    posterior = forward_inference_custom_transition_matrix(signal)

    # posterior = forward_inference_custom_transition_matrix(signal.flatten())

    # p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
    #                             false_negative=0.0, false_positive=0.0) # param_vals[1]
    # p_lick = apply_strategy(p_lick, k=param_vals[0], midpoint=param_vals[1])

    p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=0.0, false_positive=0.0) # param_vals[1]
    p_lick = apply_strategy(p_lick, k=nonstandard_sigmoid(param_vals[0], min_val=0, max_val=20), midpoint=nonstandard_sigmoid(param_vals[1], min_val=-10, max_val=10),
                            max_val=1-1e-5, min_val=1e-5)

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


def loss_function_batch(param_vals, batch):
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
    batch_loss = matrix_weighted_cross_entropy_loss(lick, batch_predictions, alpha=1)

    # adding log barrier
    # c = 10
    # barrier_loss = - c * np.sum(np.log(param_vals[num_non_hazard_rate_params:])) - c * np.sum(np.log(1 - param_vals[num_non_hazard_rate_params:]))
    # batch_loss = batch_loss + barrier_loss
    # print("Barrier loss: ", str(barrier_loss))

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

def matrix_cross_entropy_loss(lick_matrix, prediction_matrix):
    mask = np.where(lick_matrix == 99, 0, 1)
    prediction_matrix_mask = np.where(prediction_matrix == 0.88888888, 0, 1)
    cross_entropy = - ((lick_matrix * np.log(prediction_matrix)) + (1 - lick_matrix) * np.log(1-prediction_matrix))
    cross_entropy = cross_entropy * mask * prediction_matrix_mask

    return np.nansum(cross_entropy)  # nansum is just a quick fix, likely need to be more principled...


def matrix_weighted_cross_entropy_loss(lick_matrix, prediction_matrix, alpha=1):
    mask = np.where(lick_matrix == 99, 0, 1)
    prediction_matrix_mask = np.where(prediction_matrix == 0.88888888, 0, 1)
    # cross_entropy = -(alpha * lick_matrix * np.log(prediction_matrix) + (1 - lick_matrix) * np.log(1-prediction_matrix))
    # cross_entropy = cross_entropy * mask * prediction_matrix_mask

    # modifying cross entropy to prevent NaNs (probably due to underflow of predcition_matrix???)

    small_value = 1e-9
    cross_entropy = - (alpha * lick_matrix * np.log(np.maximum(prediction_matrix, small_value)) +
                      (1 - lick_matrix) * np.log(1 - np.maximum(prediction_matrix, small_value))
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

def run_through_dataset_fit_vector(datapath, savepath, training_savepath, param=None, num_non_hazard_rate_param=2):
    """
    Takes in the experimental data, and makes inference of p(lick) for each time point given the stimuli
    :param datapath:
    :param savepath:
    :param param:
    :return:
    """

    if param is None:
        # if no parameters specified, then load the training result and get the last param
        with open(training_savepath, "rb") as handle:
            training_result = pkl.load(handle)

    # normally -1 (last training epoch)
    epoch_index = 179
    param = training_result["param_val"][epoch_index]
    print("Parameter training loss: ",  str(training_result["loss"][epoch_index])) # just to double check

    # note that parameters do not need to be pre-processed

    global time_shift
    time_shift = 5


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


    signal_matrix, lick_matrix = create_vectorised_data(datapath)
    # batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    global num_non_hazard_rate_params
    num_non_hazard_rate_params = num_non_hazard_rate_param # this is for predict_lick_w_hazard_rate

    batched_predict_lick = vmap(predict_lick_w_hazard_rate, in_axes=(None, 0))


    # signals has to be specially prepared to get the vectorised code running
    prediction_matrix = batched_predict_lick(param, signal_matrix)
    # prediction_matrix[lick_matrix == 99] = onp.nan

    prediction_matrix = np.where(lick_matrix == 99, onp.nan, prediction_matrix)

    if time_shift != 0:
        # convert baseline to something sensible
        baseline_p_lick = 0.0001
        prediction_matrix = np.where(prediction_matrix == 0.88888888, baseline_p_lick, prediction_matrix)


    vec_dict = dict()
    vec_dict["change_value"] = change
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = prediction_matrix
    vec_dict["true_change_time"] = tau

    with open(savepath, "wb") as handle:
        pkl.dump(vec_dict, handle)


def control_model(datapath, savepath):

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
    # _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath)
    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="constant", datapath=datapath, constant_val=0.0001)

    global batched_predict_lick
    batched_predict_lick = vmap(predict_lick_control, in_axes=(None, 0))
    prediction_matrix = batched_predict_lick(None, signal_matrix)

    # Remove predictions after the mouse licks, since there are no real signals after the mouse licked
    prediction_matrix = np.where(lick_matrix == 99, onp.nan, prediction_matrix)

    vec_dict = dict()
    vec_dict["change_value"] = change
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = prediction_matrix
    vec_dict["true_change_time"] = tau
    vec_dict["subjective_change"] = subjective_change

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

    # print("Shape of loss", np.shape(loss))

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

def plot_training_loss(training_savepath, figsavepath=None):

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    epoch_per_step = 10
    loss = training_result["loss"]
    parameters = training_result["param_val"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(1, len(loss)+1) * epoch_per_step, loss)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


def plot_sigmoid_comparisions(training_savepath, figsavepath=None):

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    parameters = training_result["param_val"]
    num_set_to_plot = 10
    epoch_step_size = 10
    total_num_set = len(parameters)
    set_index_to_plot = onp.round(onp.linspace(0, total_num_set-1, num_set_to_plot))
    epochs_plotted = (set_index_to_plot + 1) * epoch_step_size

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


def get_hazard_rate(hazard_rate_type="subjective", datapath=None, plot_hazard_rate=False, figsavepath=None, constant_val=0.001):
    """
    The way I see it, there is 3 types of varying hazard rate.
    1. "normative": One specified by the experimental design; the actual distribution where the trial change-times are sampled from
    2. "experimental": The distribution in the trial change-times
    3. "subjective": The distribution experienced by the mice
    :param hazard_rate_type:
    :param datapath:
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


def make_transition_matrix(hazard_rate_vec):
    # standard logistic function to contrain values to [0, 1]
    # hazard_rate_vec = standard_sigmoid(hazard_rate_vec)

    # Custom logistic function to constrain values
    # small_value = 0.01
    # hazard_rate_vec = nonstandard_sigmoid(hazard_rate_vec, min_val=0, max_val=1, k=5)

    # softmax
    hazard_rate_vec = softmax(hazard_rate_vec)

    # Directly clip the values
    small_value = 1e-5
    hazard_rate_vec = np.where(hazard_rate_vec <= 0, small_value, hazard_rate_vec)
    hazard_rate_vec = np.where(hazard_rate_vec >= 1, 1-small_value, hazard_rate_vec)

    transition_matrix_list = list()
    for hazard_rate in hazard_rate_vec:
        transition_matrix = np.array([[1 - hazard_rate, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0, hazard_rate/5.0],
                                  [0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]])
        transition_matrix_list.append(transition_matrix)

    return transition_matrix_list

#################################### FITTING HAZARD RATE ETC. #########################################################

def gradient_descent_w_hazard_rate(exp_data_path, training_savepath, init_param_vals=np.array([10.0, 0.5]),
                                time_shift_list=np.arange(0, 5), num_epoch=10, fit_hazard_rate=True,
                                   n_params=2, batch_size=None):
    """

    :return:
    """
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
    hazard_rate,_ = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_path)

    # add a small vlaue so that none of them are 0 (prevent early clipping)
    small_value = 0.01
    hazard_rate = hazard_rate + small_value

    # initialise hazard rate randomly
    # hazard_rate, _ = get_hazard_rate(hazard_rate_type="random", datapath=exp_data_path)
    
    # random initialisation
    # key = random.PRNGKey(777)
    # hazard_rate_random = random.normal(key, shape=(len(hazard_rate), ))
    small_baseline = 1e-5
    hazard_rate = np.where(hazard_rate<=0, small_baseline, hazard_rate)

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
        _, transition_matrix_list = get_hazard_rate(hazard_rate_type="subjective", datapath=exp_data_path)


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

def nonstandard_sigmoid(input, min_val=0.0, max_val=1.0, k=1):
    # naive implementation
    # output = (max_val - min_val) / (1.0 + np.exp(-input)) + min_val

    # Numerically stable version
    """
    if input >= 0:
        z = np.exp(-input)
        output = ((max_val - min_val) / (1.0 + z)) + min_val
    else:
        z = np.exp(input)
        output = (z * (max_val - min_val) / (1.0 + z)) + min_val
    """

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                      (max_val - min_val) / (1.0 + np.exp(-input * k)) + min_val,
                      (max_val - min_val) * np.exp(input * k) / (1.0 + np.exp(input * k)) + min_val)

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

    # posterior = forward_inference_w_tricks(signal.flatten())
    global transition_matrix_list
    transition_matrix_list = make_transition_matrix(hazard_rate_vec=param_vals[num_non_hazard_rate_params:])
    # ^ note this works due to zero-indexing. (if num=2, then we start from index 2, which is the 3rd param)

    # add noise to the signal
    key = random.PRNGKey(777)
    signal = signal.flatten() + (random.normal(key, (len(signal.flatten()), )) * nonstandard_sigmoid(param_vals[2], min_val=0.0, max_val=1.0))
    # multiply standard normal by constant: aX + b = N(au + b, a^2\sigma^2)
    posterior = forward_inference_custom_transition_matrix(signal)

    posterior = np.where(posterior > 1, 1, posterior) # I've seen rounding errors where the output is 1.0000001 (strange)
    # Flagging just in case it as cosequences to other models... (adding this for model 40)

    # no noise in the signal
    # posterior = forward_inference_custom_transition_matrix(signal.flatten())

    p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=0.0, false_positive=0.0) # param_vals[1]
    # p_lick = apply_strategy(p_lick, k=param_vals[0], midpoint=param_vals[1])

    p_lick = apply_strategy(p_lick, k=nonstandard_sigmoid(param_vals[0], min_val=0.0, max_val=20), midpoint=nonstandard_sigmoid(param_vals[1], min_val=-10, max_val=10),
                            max_val=1, min_val=0)

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

def predict_lick_control(param, signal):
    posterior = forward_inference_custom_transition_matrix(signal.flatten())
    p_lick = np.where(posterior > 1, 1, posterior) # I've seen rounding errors where the output is 1.0000001 (strange)
    # Flagging just in case it as cosequences to other models...
    return p_lick


def plot_trained_hazard_rate(training_savepath, figsavepath, num_non_hazard_rate_param=2):
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    last_epoch_trained_param = training_result["param_val"][-1]
    hazard_rate = last_epoch_trained_param[num_non_hazard_rate_param:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.plot(standard_sigmoid(hazard_rate))
    ax.plot(softmax(hazard_rate))
    ax.set_xlabel("Time (frames")
    ax.set_ylabel("P(change)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


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


def plot_time_shift_training_result(training_savepath, figsavepath=None, time_shift_list = np.arange(0, 22, 2),
                                    num_step=20, num_epoch_per_step=10):
    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    loss = training_result["loss"]
    final_losses = loss[num_step-1::num_step] # 9th, 19th, etc... (0 indexing)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(time_shift_list, final_losses)
    ax.scatter(time_shift_list, final_losses)
    ax.set_ylabel("Loss at %s-th iteration" % str(num_step * num_epoch_per_step))
    ax.set_xlabel("Time shift (frames)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


def plot_posterior(datapath, figsavepath=None):

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"]
    change_magnitude = exp_data["sig"]
    change_time = exp_data["change"]


    ########### PLOT MEAN POSTERIOR over time GROUPED BY STIMULUS MAGNITUDE

    posterior_list = list()

    global transition_matrix_list
    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath)


    # batched_get_posterior = vmap(forward_inference_custom_transition_matrix, in_axes=(None, 0))
    # posterior_array = batched_get_posterior(signals)

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


def gradient_clipping(gradients, type="L2_norm"):
    if type == "L2_norm":
        threshold = 10 # need to read what to set this to...
        new_gradients = gradients * threshold / (gradients ** 2 / len(gradients)) # TODO: sohoudl be l2 norm

    return new_gradients


def main(model_number=99,run_test_foward_algorithm=False, run_test_on_data=False, run_gradient_descent=False, run_plot_training_loss=False, run_plot_sigmoid=False, run_plot_test_loss=False,
         run_model=False, run_plot_time_shift_test=False, run_plot_hazard_rate=False, run_plot_trained_hazard_rate=False,
         run_benchmark_model=False, run_plot_time_shift_training_result=False, run_plot_posterior=False, run_control_model=False):

    print("Running model: ", str(model_number))

    # datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/subsetted_data/data_IO_083.pkl"
    exp_data_number = 83
    main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    # TODO: generalise the code below
    datapath = os.path.join(main_folder, "exp_data/subsetted_data/data_IO_083.pkl")
    model_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/model_response_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".pkl")
    fig_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/model_response_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".png")
    training_savepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/training_result_0" + str(exp_data_number) + "_"
                                   + str(model_number) + ".pkl")

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
    # simple_gradient_descent(datapath, num_epoch=100, param_vals = np.array([9.59, 0.27, 0.37]),
    #                         training_savepath=training_savepath)
    # plot_training_result(training_savepath)

    if run_model is True:
        run_through_dataset_fit_vector(datapath=datapath, savepath=model_save_path, training_savepath=training_savepath,
                                       num_non_hazard_rate_param=3,
                                       param=None)

    if run_control_model is True:
        control_model(datapath=datapath, savepath=model_save_path)

    # compare_model_with_behaviour(model_behaviour_df_path=model_save_path, savepath=fig_save_path, showfig=True)

    # gradient_descent_fit_vector(exp_data_path=datapath,
    #                             training_savepath=training_savepath,
    #                             init_param_vals=np.array([10.0, 0.0, 0.5]),
    #                             time_shift_list=np.arange(0, 5), num_epoch=10)

    if run_gradient_descent is True:
        # gradient_descent_fit_vector_faster(exp_data_path=datapath,
        #                                     training_savepath=training_savepath,
        #                                     init_param_vals=np.array([10.0, 0.5]),
        #                                     time_shift_list=np.arange(0, 102, 2), num_epoch=100)

        gradient_descent_w_hazard_rate(exp_data_path=datapath,
                                            training_savepath=training_savepath,
                                            init_param_vals=np.array([0.0, 0.1, 0.1]), # originally 10.0, 0.5, 0.1
                                            n_params=3, fit_hazard_rate=True,
                                           time_shift_list=np.arange(5, 10), num_epoch=200, batch_size=512)
        # batch size originally 128

        # sim 32:  time_shift_list=np.arange(0, 22, 2), num_epoch=200




    # test_vectorised_inference(exp_data_path=datapath)

    if run_plot_test_loss is True:
        figsavepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/test_alpha_model_" + str(model_number) + ".png")
        test_loss_function(datapath, plot_example=False, savepath=figsavepath)

    if run_plot_training_loss is True:
        figsavepath = os.path.join(main_folder, "figures/training_result_model" + str(model_number))
        plot_training_loss(training_savepath, figsavepath=figsavepath)

    if run_plot_sigmoid is True:
        figsavepath = os.path.join(main_folder, "figures/transfer_function_model" + str(model_number))
        plot_sigmoid_comparisions(training_savepath, figsavepath=figsavepath)

    if run_plot_trained_hazard_rate is True:
        figsavepath = os.path.join(main_folder, "figures/trained_hazard_rate" + str(model_number))
        plot_trained_hazard_rate(training_savepath, figsavepath=figsavepath, num_non_hazard_rate_param=3)


    if run_plot_time_shift_test is True:
        for trial_num in np.arange(0, 5):
            plot_time_shift_test(datapath, param=[10, 0.5], time_shift_list=[0, -10], trial_num=trial_num)

    if run_plot_hazard_rate is True:
        figsavepath = os.path.join(main_folder, "figures/hazard_rate_subjective_mouse_" + str(exp_data_number) + ".png")
        get_hazard_rate(hazard_rate_type="subjective", datapath=datapath, plot_hazard_rate=True,
                        figsavepath=figsavepath)

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
        plot_posterior(datapath=datapath, figsavepath=None)



if __name__ == "__main__":
    main(model_number=1002, run_test_on_data=False, run_gradient_descent=False, run_plot_training_loss=False,
         run_plot_sigmoid=False,
         run_plot_test_loss=False, run_model=False, run_plot_time_shift_test=False,
         run_plot_hazard_rate=False, run_plot_trained_hazard_rate=False, run_benchmark_model=False,
         run_plot_time_shift_training_result=False, run_plot_posterior=False, run_control_model=True)

