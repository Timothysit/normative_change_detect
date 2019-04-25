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


def apply_strategy(prob_lick, k, midpoint=0.5):
    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    # two param logistic function
    max_val = 0.99  # To prevent 1.0
    p_lick = max_val / (1 + np.exp(-k * (prob_lick - midpoint)))

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
        trial_index = np.where(change_magnitude == change_val)[0]
    else:
        trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0))[0]

    signal = exp_data["ys"].flatten()[trial_index][0][0]
    tau = exp_data["change"][trial_index][0][0]

    # print("Signal baseline mean:", )

    p_z_given_x = forward_inference(signal)

    # Plot example
    plot_signal_and_inference(signal=signal, tau=tau, prob=p_z_given_x)

    return None


def posterior_to_decision(posterior, return_prob=True, k=1.0, false_negative=0.0):
    change_posterior = posterior  # p(z_2 | x_{1:k})
    p_lick = apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=false_negative, false_positive=0.0)
    p_lick = apply_strategy(p_lick, k=k)

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
    global actual_lick_vector
    global signals
    actual_lick_vector = mouse_lick_vector_list
    signals = signal

    global time_shift

    loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))
    final_param_list = list() # list of list, each list is the parameter values for a particular time shift

    loss_val_list = list()

    for time_shift in time_shift_list:

        step_size = 0.001
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





        final_param_list.append(trained_params)

    training_result = dict()
    training_result["loss"] = loss_val_matrix
    training_result["param_val"] = final_param_list

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def create_vectorised_data(exp_data_path):
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
            mouse_lick_vector[int(mouse_rt[trial] - 1)] = 1
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
    signal_matrix, lick_matrix = create_vectorised_data(exp_data_path)

    init_param_vals = np.array([11.0, 0.0, 0.5])

    # TODO: Think about how to do time shift

    global time_shift

    # loss_val_matrix = onp.zeros((num_epoch, len(time_shift_list)))
    param_list = list() # list of list, each list is the parameter values for a particular time shift
    loss_val_list = list()
    # define batched prediction function using vmap
    global batched_predict_lick
    batched_predict_lick = vmap(predict_lick, in_axes=(None, 0))

    # print("Initial parameters:", init_param_vals)

    config.update("jax_debug_nans", True) # nan debugging

    for time_shift in tqdm(time_shift_list):

        step_size = 0.001
        momentum_mass = 0.9
        opt_init, opt_update = optimizers.momentum(step_size, mass=momentum_mass)

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

        # TODO: Implement minibatch gradient descent
        num_train = onp.shape(signal_matrix)[0]
        batch_size = 128
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
        def step(i, opt_state):
            params = optimizers.get_params(opt_state)
            # print("Params within step:", params)
            g = grad(loss_function_fit_vector_faster)(params)
            return opt_update(i, g, opt_state)

        for epoch in range(num_epoch):
            opt_state = step(epoch, opt_state)
            params = optimizers.get_params(opt_state)
            print("Params outside step", params)
            loss_val = loss_function_fit_vector_faster(params)
            print("Loss:", loss_val)
            loss_val_list.append(loss_val)
            param_list.append(params)
            # print("Parameters:", params)



        # Simple Gradient Descent
        """
        params = init_param_vals
        for epoch in range(num_epoch):
            print(epoch, params)
            gradient = grad(loss_function_fit_vector_faster)(params)
            print(gradient)
            params -= 0.001 * gradient
            loss = loss_function_fit_vector_faster(params)
            print("Loss: ", loss)
            loss_val_list.append(loss)
            param_list.append(params)
        """

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = param_list

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def predict_lick(param_vals, signal):
    # posterior = forward_inference(signal.flatten())
    posterior = forward_inference_w_tricks(signal.flatten())
    p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=param_vals[1], false_positive=0.0)
    p_lick = apply_strategy(p_lick, k=param_vals[0], midpoint=param_vals[2])
    baseline = 0.01
    # Add time shift
    # p_lick = np.concatenate([np.repeat(baseline, time_shift), p_lick])
    # p_lick = p_lick[0:len(actual_lick_vector)]  # clip p(lick) to be same length as actual lick

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
    :return:
    """

    batch_predictions = batched_predict_lick(param_vals, signal_matrix)
    # batch_loss = batched_cross_entropy(actual_lick_vector=lick_matrix, p_lick=batch_predictions)
    batch_loss = matrix_cross_entropy_loss(lick_matrix, batch_predictions)

    return batch_loss


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
    cross_entropy = -(lick_matrix * np.log(prediction_matrix)) + ((1 - lick_matrix) * np.log(1-prediction_matrix))
    cross_entropy = cross_entropy * mask

    return np.nansum(cross_entropy)  # nansum is just a quick fix, likely need to be more principled...


def matrix_weighted_cross_entropy_loss(lick_matrix, prediction_matrix, alpha=1):
    mask = np.where(lick_matrix == 99, 0, 1)
    cross_entropy = -(alpha * lick_matrix * np.log(prediction_matrix)) + ((1 - lick_matrix) * np.log(1-prediction_matrix))
    cross_entropy = cross_entropy * mask

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

def run_through_dataset_fit_vector(datapath, savepath, param):
    """
    Takes in the experimental data, and makes inference of p(lick) for each time point given the stimuli
    :param datapath:
    :param savepath:
    :param param:
    :return:
    """

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"].flatten()
    # mouse_rt = exp_data["rt"].flatten()
    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()
    absolute_decision_time = exp_data["rt"].flatten()
    # peri_stimulus_rt = absolute_decision_time - tau

    # Experimental data
    # mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    # mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    # mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()

    lick_choice_vector = list()

    for signal in tqdm(signals):
        signal = signal.reshape(-1, 1)

        posterior = forward_inference(signal.flatten())

        lick_choice = posterior_to_decision(posterior, return_prob=True, k=param[0], false_negative=param[1],
                                            midpoint=param[2])

        lick_choice_vector.append(lick_choice)

    # TODO: Add time shift

    vec_dict = dict()
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = lick_choice_vector
    vec_dict["true_change_time"] = tau

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

def test_loss_function(datapath):

    params = [10.0, 0.0, 0.5]
    time_shift = 0

    signal_matrix, lick_matrix = create_vectorised_data(datapath)


    # Plot examples
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




def main():
    # datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/subsetted_data/data_IO_083.pkl"
    datapath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/exp_data/subsetted_data/data_IO_083.pkl")
    model_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/model_response_083_21.pkl")
    fig_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/model_response_083_21.png")
    training_savepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/training_result_083_21.pkl")

    """
    param: 
    1. k parameter in the sigmoid function
    2. false_negative value in cost-benefit function
    3. midpoint of the sigmoid decision function
    """

    # test_forward_algorithm()
    # test_on_data(change_val=1.35, exp_data_file=datapath)
    # simple_gradient_descent(datapath, num_epoch=100, param_vals = np.array([9.59, 0.27, 0.37]),
    #                         training_savepath=training_savepath)
    # plot_training_result(training_savepath)
    # run_through_entire_dataset(datapath=datapath, savepath=model_save_path, param=[1000, 0, 0.5], numtrial=None)
    # compare_model_with_behaviour(model_behaviour_df_path=model_save_path, savepath=fig_save_path, showfig=True)

    # gradient_descent_fit_vector(exp_data_path=datapath,
    #                             training_savepath=training_savepath,
    #                             init_param_vals=np.array([10.0, 0.0, 0.5]),
    #                             time_shift_list=np.arange(0, 5), num_epoch=10)

    # gradient_descent_fit_vector_faster(exp_data_path=datapath,
    #                                 training_savepath=training_savepath,
    #                                 init_param_vals=np.array([10.0, 0.0, 0.5]),
    #                                 time_shift_list=np.arange(0, 5), num_epoch=10)

    # test_vectorised_inference(exp_data_path=datapath)


    test_loss_function(datapath)



if __name__ == "__main__":
    main()

