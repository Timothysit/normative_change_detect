# Automatic Differentiation
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

# Other things
import hmm  # functions to simulate data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from tqdm import tqdm

import os
import pickle as pkl
import pandas as pd

from os.path import expanduser
home = expanduser("~")

# Testing
from hmm2_jax import matrix_weighted_cross_entropy_loss
from hmm2_jax import predict_lick
from hmm2_jax import create_vectorised_data


def cal_p_x_given_z1(x_k):
    # calculates the probability of the obv. given the baseline state
    z1_mu = np.log(1.0)
    z1_var = 0.25  # sigma^2

    p_x_given_z1 = (1 / np.sqrt(2 * np.pi * z1_var)) * np.exp(-(x_k - z1_mu) ** 2 / (2 * z1_var))

    return p_x_given_z1


def cal_p_x_given_z2(x_k):
    # calcaultes the probability of the obv. given the change state
    z2_mu = np.log(np.array([1.25, 1.35, 1.50, 2.00, 4.00]))
    z2_var = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
    z2_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # z2_mu = 1.0
    # z2_var = 0.25

    p_x_given_z2 = (1 / np.sqrt(2 * np.pi * z2_var)) * np.exp(-(x_k - z2_mu) ** 2 / (2 * z2_var))
    # p_x_given_z2 = np.dot(p_x_given_z2, z2_weights)
    p_x_given_z2 = np.max(p_x_given_z2)

    return p_x_given_z2

def cal_p_x_given_z(x_k):
    z_mu = np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00]))
    z_var = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2 # in actual data, std is 0.25

    # z_mu = np.log(np.array([1.0, 1.25]))
    # z_var = np.array([0.25, 0.25])

    p_x_given_z = (1 / np.sqrt(2 * np.pi * z_var)) * np.exp(-(x_k - z_mu) ** 2 / (2 * z_var))

    # returns row vector (for simpler calculation later on)
    return p_x_given_z.T

def cal_p_x_given_z_custom(x_k, z_mu=np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00])),
                           z_var=np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2):
    p_x_given_z = (1 / np.sqrt(2 * np.pi * z_var)) * np.exp(-(x_k - z_mu) ** 2 / (2 * z_var))
    # returns row vector (for simpler calculation later on)
    return p_x_given_z.T


def cal_p_x_given_zchange(x_k, z_n):
    zchange_mu_list = np.log(np.array([1.25, 1.35, 1.50, 2.00, 4.00]))
    zchange_var_list = np.array([0.25, 0.25, 0.25, 0.25, 0.25])

    zchange_mu = zchange_mu_list[z_n]
    zchange_var = zchange_var_list[z_n]

    p_x_given_zchange = (1 / np.sqrt(2 * np.pi * zchange_var)) * np.exp(-(x_k - zchange_mu) ** 2 / (2 * zchange_var))

    # returns row vector
    return p_x_given_zchange.T


"""
def forward_inference(x):
    # TODO: Think about logging thing to prevent underflow
    p_z_given_x = np.zeros((len(x), 2))
    p_z_and_x = np.zeros((len(x), 2))

    hazard_rate = 0.0001
    transition_matrix = np.array([[1 - hazard_rate, hazard_rate], [0, 1]])
    init_state_probability = np.array([1.0, 0.0])

    # Iniitial probabilities
    p_z_and_x[0, 0] = cal_p_x_given_z1(x_k=x[0]) * init_state_probability[0]
    p_z_and_x[0, 1] = cal_p_x_given_z2(x_k=x[0]) * init_state_probability[1]

    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_z_and_x[k, 0] = cal_p_x_given_z1(x_k=x[k]) * transition_matrix[0, 0] * p_z_and_x[k-1, 0] + \
                          cal_p_x_given_z2(x_k=x[k]) * transition_matrix[1, 0] * p_z_and_x[k-1, 1]
        p_z_and_x[k, 1] = cal_p_x_given_z1(x_k=x[k]) * transition_matrix[0, 1] * p_z_and_x[k-1, 0] + \
                           cal_p_x_given_z2(x_k=x[k]) * transition_matrix[1, 1] * p_z_and_x[k-1, 1]
        # p_z_and_x[k, 1] = 1 - p_z_and_x[k, 0]

        # NOTE: This step is not the conventional forward algorithm, but works.
        p_x = p_z_and_x[k, 0] + p_z_and_x[k, 1]
        p_z_given_x[k, 0] = p_z_and_x[k, 0] / p_x # p(x_{1:k})
        p_z_given_x[k, 1] = p_z_and_x[k, 1] / p_x

    return p_z_given_x
"""


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
        p_xk_given_z_store[k, :] = p_xk_given_z

        # update joint
        p_z_and_x = np.dot((p_xk_given_z * p_z_and_x), transition_matrix)

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)

        p_change_given_x.append(np.sum(p_zk_given_xk[1:])) # sum from the second element
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


def apply_strategy(prob_lick, k, midpoint=0.5, baseline=0):
    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    # baseline offset
    # avoiding outputting zero probability (will make cross entropy impossible to calculate)

    # two param logistic function
    max_val = 1 - baseline  # L
    p_lick = max_val / (1 + np.exp(-k * (prob_lick - midpoint))) + baseline

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


def loss_function(param_vals):
    # missing arguments: signals, param_vals_, actual_lick_,
    """
    loss between predicted licks and actual licks
    computed using cross entropy
    This loss function is for licking versus no-licking, but does not compute the loss of the reaction time.
    :return:
    """

    cross_entropy_loss = 0
    for x_num, x in enumerate(signals):
        posterior = forward_inference(x.flatten())
        p_lick = apply_cost_benefit(change_posterior=posterior, true_positive=1.0, true_negative=0.0,
                                    false_negative=0.0, false_positive=0.0)
        p_lick = apply_strategy(p_lick, k=param_vals[0])
        p_lick = np.max(p_lick)
        cross_entropy_loss += -(actual_lick[x_num] * np.log(p_lick) + (1 - actual_lick[x_num]) * np.log(
            1 - p_lick))

    return cross_entropy_loss


def loss_function_iter(param_vals, iter):
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
        p_lick = np.max(p_lick)
        cross_entropy_loss += -(actual_lick[x_num] * np.log(p_lick) + (1 - actual_lick[x_num]) * np.log(
            1 - p_lick))

    return cross_entropy_loss


# Test forward algorithm
def test_forward_algorithm():
    x = hmm.sim_data(u_1=0.0, u_2=1.0, sigma=0.25, tau=50, n=100, update_interval=1, dist="normal",
                     noise_mean=None, noise_sigma=None, noise_dist="log_normal")

    joint_prob = forward_inference(x)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(x)
    axs[1].plot(joint_prob[:, 1])

    plt.show()


def plot_signal_and_inference(signal, tau, prob, savepath=None):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(signal)
    axs[0].axvline(tau, color="r", linestyle="--", linewidth=2.0)

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


def posterior_to_decision(posterior, return_prob=True, k=1.0, false_negative=0.0, midpoint=0.5):
    change_posterior = posterior  # p(z_2 | x_{1:k})
    p_lick = apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=0.0,
                                false_negative=false_negative, false_positive=0.0)
    p_lick = apply_strategy(p_lick, k=k, midpoint=midpoint)

    if return_prob is True:
        return p_lick


def run_through_entire_dataset(datapath, savepath, param, numtrial=100):
    # note this will also include experimental data, so perhaps "model_behaviour" is not the best variable name...
    # runs forward algorithm through entire dataset
    # exp_data = scipy.io.loadmat(datapath)
    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"].flatten()
    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()
    absolute_decision_time = exp_data["rt"].flatten()
    peri_stimulus_rt = absolute_decision_time - tau

    # Experimental data
    mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()

    """
    if "abort" in subset_criteria:
        no_abort_index = np.where(exp_data["outcome"].flatten() != "abort")[0]
        signals = signals[no_abort_index]
        change = change[no_abort_index]
        tau = tau[no_abort_index]
    """

    decision = list()
    decision_time = list()
    lick_choice_list = list()

    for signal in tqdm(signals[0:numtrial]):
        signal = signal.reshape(-1, 1)

        posterior = forward_inference(signal.flatten())

        lick_choice = posterior_to_decision(posterior, return_prob=True, k=param[0], false_negative=param[1],
                                            midpoint=param[2])
        lick_choice = np.max(lick_choice)

        # save data
        # print(lick_choice)
        decision.append(lick_choice)  # 0 = no lick, 1 = lick
        # print(decision)

        """
        if np.max(lick_choice) == 1:
            decision_time.append(np.where(lick_choice == 1)[0][0]) # only use first lick (subsequent licks don't count)
        else:
            decision_time.append(np.nan)
        """

        # lick_choice_list.append(lick_choice)

    # TODO: Deal with abort subset criteria below (lengths not equal)

    # model_response = dict()
    # model_response["decision"] = decision
    # model_response["decision_time"] = decision_time

    # save the behaviour
    model_behaviour = dict()
    model_behaviour["change"] = change[0:numtrial]
    model_behaviour["decision"] = decision[0:numtrial]
    model_behaviour["tau"] = tau[0:numtrial]
    model_behaviour["mouse_lick"] = mouse_lick[0:numtrial]
    model_behaviour["mouse_decision_time"] = absolute_decision_time[0:numtrial]

    # model_behaviour["lick_choice_list"] = lick_choice_list

    # Convert from dictionary to dataframe
    model_behaviour_df = pd.DataFrame.from_dict(model_behaviour)

    with open(savepath, "wb") as handle:
        pkl.dump(model_behaviour_df, handle)


def simple_gradient_descent(datapath, training_savepath, param_vals=np.array([10.0, 0.0, 0.5]), num_epoch=100):
    """
    parameters:
    0: k (curve of the sigmoid function)
    1: false_negative
    2: midpoint (sigmoid function)
    """
    # Define targets (actual licks)
    # exp_data_file = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/data/data_IO_083.mat"
    # exp_data = scipy.io.loadmat(exp_data_file)

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    reaction_time = exp_data["rt"]
    mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()

    signal = exp_data["ys"].flatten()

    # subset trials
    # noiseless_trial_type = exp_data["noiseless"].flatten()
    # mouse_abort = (exp_data["outcome"].flatten() == "abort").astype(float)
    # trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0))[0]
    # mouse_lick = mouse_lick[trial_index]
    # signal = signal[trial_index]

    # only get first 100 trials just as a test
    # mouse_lick = mouse_lick[0:100]
    # signal = signal[0:100]

    # Define global variables used by loss_function
    global actual_lick
    global signals
    actual_lick = mouse_lick
    signals = signal

    # loss_value = loss_function(actual_lick=mouse_lick, signals=signal, param_vals=param_vals)

    # Simple Gradient Descent
    loss_function_grad = grad(loss_function)

    learning_rate = 0.01

    # print("Initial gradient:", loss_function_grad(param_vals))

    loss_val_list = list()
    num_param = len(param_vals)
    param_val_array = np.zeros((num_epoch, num_param))

    # simple gradient descent:
    """
    for i in tqdm(range(num_epoch)):
        loss_val_list.append(loss_function_grad(param_vals))
        param_val_array[i, :] = param_vals
        param_vals -= loss_function_grad(param_vals) * learning_rate
    """

    # ADAM Optimisation
    loss_function_grad = grad(loss_function_iter)

    def callback(weights, iter, gradient):
        if iter % 5 == 0:
            train_loss = loss_function_iter(weights, 0)
            print("Iteration", iter, "Train loss:", train_loss)
            print("Gradient", gradient)
            loss_val_list.append(train_loss)

            # TODO: Save the weights as well.

    trained_params = adam(loss_function_grad, param_vals, step_size=0.01,
                          num_iters=num_epoch, callback=callback)
    param_val_array = trained_params

    # print("Trained loss:", loss_function_grad(param_vals))

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = param_val_array

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


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
    for trial in np.arange(0, len(mouse_rt)):
        if not np.isnan(mouse_rt[trial]):
            mouse_lick_vector = np.zeros(shape=(int(mouse_rt[trial]), ))
            mouse_lick_vector[int(mouse_rt[trial] - 1)] = 1
            mouse_lick_vector_list.append(mouse_lick_vector)
        else:
            mouse_lick_vector = np.zeros(shape=(len(signal[trial][0])), )
            mouse_lick_vector_list.append(mouse_lick_vector)

    assert len(signal) == len(mouse_lick_vector_list) # check equal number of trials

    signal = exp_data["ys"].flatten()

    # Define global variables used by loss_function
    global actual_lick_vector
    global signals
    actual_lick_vector = mouse_lick_vector_list
    signals = signal

    global time_shift

    # loss_val_matrix = np.zeros((num_epoch, len(time_shift_list)))
    loss_val_list = list()
    final_param_list = list() # list of list, each list is the parameter values for a particular time shift


    # callback function to get loss and parametervalues
    def callback(weights, iter, gradient):
        if iter % 5 == 0:
            train_loss = loss_function_fit_vector(weights, 0)
            print("Iteration", iter, "Train loss:", train_loss)
            print("Gradient", gradient)
            loss_val_list.append(train_loss)

    # print("testing loss function")
    for time_shift in time_shift_list:
        param_vals = init_param_vals
        loss_function_fit_vector(param_vals, iter=0)


    for time_shift in time_shift_list:
        param_vals = init_param_vals # reset initial parameter values for each gradient descent.
        loss_val_list = list()
        # ADAM Optimisation
        loss_function_grad = grad(loss_function_fit_vector)
                # TODO: Save the weights as well.

        trained_params = adam(loss_function_grad, param_vals, step_size=0.01,
                              num_iters=num_epoch, callback=callback)

        final_param_list.append(trained_params)

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = final_param_list
    training_result["time_shift"] = time_shift_list

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def loss_function_fit_vector(param_vals, iter):
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

    # TODO: Add reaction time to the comparison

    # Plot model behaviour
    ax.plot(model_prop_choice.change, model_prop_choice.decision)
    ax.scatter(model_prop_choice.change, model_prop_choice.decision)

    # Plot mouse behaviour
    ax.plot(mouse_prop_choice.change, mouse_prop_choice.mouse_lick)
    ax.scatter(mouse_prop_choice.change, mouse_prop_choice.mouse_lick)

    ax.legend(["Model", "Mouse"], frameon=False)

    ax.set_xlabel("Change magnitude")
    ax.set_ylabel("P(lick)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300)

    if showfig is True:
        plt.show()


def run_through_dataset_fit_vector(datapath, savepath, param, time_shift=0):
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

        # Introduce time shift
        baseline = 0.01
        lick_choice = np.concatenate([np.repeat(baseline, time_shift), lick_choice])

        lick_choice_vector.append(lick_choice)

    # TODO: Add time shift

    vec_dict = dict()
    vec_dict["rt"] = absolute_decision_time
    vec_dict["model_vec_output"] = lick_choice_vector
    vec_dict["true_change_time"] = tau
    vec_dict["change_value"] = change
    vec_dict["block_type"] = exp_data["hazard"].flatten()

    with open(savepath, "wb") as handle:
        pkl.dump(vec_dict, handle)



def plot_fit_vector_performance(model_save_path, figsavepath=None):

    with open(model_save_path, "rb") as handle:
        model_response = pkl.load(handle)

    mouse_rt = model_response["rt"]
    model_prediction = model_response["model_vec_output"]
    true_change_time = model_response["true_change_time"]

    # Align with mouse lick
    fig, ax = plt.subplots(figsize=(8, 6))
    window_width = 100
    for trial, rt in enumerate(mouse_rt):
        if not np.isnan(rt):
            start_time = int(rt - window_width/2)
            end_time = int(rt + window_width/2)
            trace = model_prediction[trial][start_time:end_time]
            ax.plot(np.arange(0, len(trace)) - window_width/2,
                    trace, alpha=0.1, color="blue")

        ax.axvline(0, linestyle="--", color="green")

    ax.set_ylabel("P(lick)")
    ax.set_xlabel("Peri-lick time (frames)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath)

    plt.show()

    # Align with actual change time
    fig, ax = plt.subplots(figsize=(8, 6))
    window_width = 50
    for trial, rt in enumerate(true_change_time):
        if not np.isnan(rt):
            start_time = int(rt - window_width/2)
            end_time = int(rt + window_width/2)
            trace = model_prediction[trial][start_time:end_time]
            ax.plot(np.arange(0, len(trace)) - window_width/2,
                    trace, alpha=0.1, color="blue")

        ax.axvline(0, linestyle="--", color="green")

    plt.show()


    # Test out 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # re-order mouse_rt and model_prediction
    time_to_rise_list = list()
    threshold = 0.7
    for p_lick_vec in model_prediction:
        time_to_rise = np.where(p_lick_vec > threshold)[0]
        if len(time_to_rise) == 0:
            time_to_rise_list.append(0)
        else:
            time_to_rise_list.append(time_to_rise[0])
    # new_sort_list = np.array(sorted(time_to_rise_list))
    new_sort_index = np.argsort(time_to_rise_list)


    window_width = 100
    """
    for trial, rt in enumerate(mouse_rt):
        if not np.isnan(rt):
            start_time = int(rt - window_width/2)
            end_time = int(rt + window_width/2)
            trace = model_prediction[trial][start_time:end_time]
            ax.plot(xs=np.arange(0, len(trace)) - window_width/2,
                    zs=trace, ys=np.repeat(trial, len(trace)), alpha=0.1, color="blue")

        # ax.axvline(0, linestyle="--", color="green")
    """
    # new loop using sorted index
    num_trace = 0
    for trial, index in enumerate(new_sort_index):
        rt = mouse_rt[index]
        model_vec = model_prediction[index]
        if not np.isnan(rt):
            start_time = int(rt - window_width/2)
            end_time = int(rt + window_width/2)
            trace = model_vec[start_time:end_time]
            ax.plot(xs=np.arange(0, len(trace)) - window_width/2,
                    zs=trace, ys=np.repeat(trial, len(trace)), alpha=0.1, color="blue")
            num_trace = num_trace + 1

    print("Number of trials with licks:", num_trace)
    plt.show()



def test_loss_function(datapath, plot_example=True, savepath=None):

    params = [10.0, 0.5, 0.0]
    time_shift = 0

    signal_matrix, lick_matrix = create_vectorised_data(datapath)


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

    all_zero_prediction_matrix = np.zeros(shape=np.shape(lick_matrix))
    all_zero_prediction_matrix[:] = 1e-8

    all_one_prediction_matrix = np.zeros(shape=np.shape(lick_matrix))
    all_one_prediction_matrix[:] = 1 - 1e-8


    alpha_list = [1, 5, 10, 20]
    # num_model = 3
    # loss_store = np.zeros(shape=(num_model, len(alpha_list)))
    all_zero_loss_store = list()
    all_one_loss_store = list()
    model_loss_store = list()

    prediction_matrix = predict_lick(params, signal_matrix)

    for n, alpha in tqdm(enumerate(alpha_list)):
        all_zero_loss = matrix_weighted_cross_entropy_loss(lick_matrix, all_zero_prediction_matrix, alpha=alpha)
        all_one_loss = matrix_weighted_cross_entropy_loss(lick_matrix, all_one_prediction_matrix, alpha=alpha)
        model_loss = matrix_weighted_cross_entropy_loss(lick_matrix, prediction_matrix, alpha=alpha)

        all_zero_loss_store.append(all_zero_loss)
        all_one_loss_store.append(all_one_loss)
        model_loss_store.append(model_loss)
        # loss_store[0, n] = all_zero_loss
        # loss_store[1, n] = all_one_loss
        # loss_store[2, n] = model_loss

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    # axs.plot(loss_store)
    axs.plot(all_zero_loss_store)
    axs.plot(all_one_loss_store)
    axs.plot(model_loss_store)
    axs.legend(["All zero", "All one", "Model"], frameon=False)

    axs.set_ylabel("Cross entropy loss")
    axs.set_xlabel("False negative bias (alpha)")

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()




def get_hazard_rate(hazard_rate_type="subjective", datapath=None, plot_hazard_rate=False, figsavepath=None):
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

    experimental_hazard_rate = np.histogram(change_time, range=(0, max_signal_length), bins=max_signal_length)[0]
    experimental_hazard_rate = experimental_hazard_rate / num_trial

    if hazard_rate_type == "subjective":
        # get change times
        # remove change times where the mice did a FA, or a miss
        outcome = exp_data["outcome"].flatten()
        hit_index = onp.where(outcome == "Hit")[0]
        num_subjective_trial = len(hit_index)
        subjective_change_time = change_time[hit_index]
        subjective_hazard_rate = \
        onp.histogram(subjective_change_time, range=(0, max_signal_length), bins=max_signal_length)[0]

        # convert from count to proportion (Divide by num_trial or num_subjective_trial?)
        subjective_hazard_rate = subjective_hazard_rate / num_subjective_trial

        assert len(subjective_hazard_rate) == max_signal_length

        hazard_rate_vec = subjective_hazard_rate

    elif hazard_rate_type == "experimental":
        hazard_rate_vec = experimental_hazard_rate
    elif hazard_rate_type == "normative":
        pass
    elif hazard_rate_type == "constant":
        hazard_rate_constant = 0.001
        hazard_rate_vec = np.repeat(hazard_rate_constant, max_signal_length)

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
        transition_matrix = np.array([[1 - hazard_rate, hazard_rate / 5.0, hazard_rate / 5.0, hazard_rate / 5.0,
                                       hazard_rate / 5.0, hazard_rate / 5.0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1]])
        transition_matrix_list.append(transition_matrix)

    return hazard_rate_vec, transition_matrix_list

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
    p_z_given_x = np.zeros(
        (len(x), 6))  # this will be a n x M matrix, not sure if this is can be created without array assignment..
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
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix_list[k - 1])

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:]))  # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    return np.array(p_change_given_x)

def forward_inference_w_noise(x, observational_noise_mean=None, observational_noise_std=None,
                              std_estimate_noise=None, mean_estimate_noise=None):
    """
    :param transtiion_matrix_list: global variable with list of transition matrices
    :param x: signal to predict
    :return:
    """

    if (observational_noise_mean is not None) and (observational_noise_std is not None):
        observational_noise = np.random.normal(loc=observational_noise_mean, scale=observational_noise_std,
                                               size=len(x))
        x = x + observational_noise


    # TODO Quite sure below can be simplified by setting the scale to 0?
    if mean_estimate_noise is not None:
        mean_estimate = np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00])) + np.random.normal(loc=0, scale=mean_estimate_noise,
                                                                                                 size=6)
    else:
        mean_estimate = np.log(np.array([1.0, 1.25, 1.35, 1.50, 2.00, 4.00]))
    if std_estimate_noise is not None:
        std_estimate = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2 + np.random.normal(loc=0, scale=std_estimate_noise)
    else:
        std_estimate = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) ** 2


    p_z1_given_x = list()
    p_z2_given_x = list()

    init_state_probability = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # List to store the posterior: p(z_k | x_{1:k}) for k = 1, ..., n
    p_z_given_x = np.zeros(
        (len(x), 6))  # this will be a n x M matrix, not sure if this is can be created without array assignment..
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
        # noisless case
        if (mean_estimate_noise is None) and (std_estimate_noise is None):
            p_xk_given_z = cal_p_x_given_z(x_k=x[k])
        elif (mean_estimate_noise is not None) and (std_estimate_noise is None):
            # noisy estimate that changes each trial (but stay the same frames for each trial)
            p_xk_given_z = cal_p_x_given_z_custom(x_k=x[k], z_mu=mean_estimate, z_var=std_estimate)

        # update conditional probability
        p_z_and_x = np.dot((p_xk_given_z * p_z_given_x), transition_matrix_list[k - 1])

        # NOTE: This step is not the conventional forward algorithm, but works.
        # p_z_given_x[k, :] = p_z_and_x / np.sum(p_z_and_x)
        # p_zk_given_xk = p_z_and_x / np.sum(p_z_and_x)
        p_z_given_x = np.divide(p_z_and_x, np.sum(p_z_and_x))

        p_change_given_x.append(np.sum(p_z_given_x[1:]))  # sum from the second element
        # p_baseline_given_x.append(p_z_given_x[0]) # Just 1 - p_change

    return np.array(p_change_given_x)




def get_posterior(datapath, posteriorsavepath=None, noisy=False):

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    signals = exp_data["ys"]
    change_magnitude = exp_data["sig"]
    change_time = exp_data["change"]


    ########### PLOT MEAN POSTERIOR over time GROUPED BY STIMULUS MAGNITUDE

    posterior_list = list()

    global transition_matrix_list
    # _, transition_matrix_list = get_hazard_rate(hazard_rate_type="experimental", datapath=datapath)
    _, transition_matrix_list = get_hazard_rate(hazard_rate_type="constant", datapath=datapath)


    # batched_get_posterior = vmap(forward_inference_custom_transition_matrix, in_axes=(None, 0))
    # posterior_array = batched_get_posterior(signals)

    for signal in tqdm(signals):
        if noisy is False:
            posterior = forward_inference_custom_transition_matrix(signal[0].flatten())
        else:
            # observational noise
            # posterior = forward_inference_w_noise(x=signal[0].flatten(), observational_noise_mean=0,
            #                                   observational_noise_std=0.5)
            # model estimate noise
            posterior = forward_inference_w_noise(x=signal[0].flatten(), observational_noise_mean=None,
                                              observational_noise_std=None, mean_estimate_noise=0.05,
                                              std_estimate_noise=None)


        posterior_list.append(posterior)


    if posteriorsavepath is not None:
        with open(posteriorsavepath, "wb") as handle:
            pkl.dump(posterior_list, handle)

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[:] = np.nan # fill w/ nan instead.

    out[mask] = np.concatenate(data)
    return out

def plot_posterior(datapath, posteriorsavepath, figsavepath=None, window_length=20):

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    change_magnitude = exp_data["sig"].flatten()
    change_time = exp_data["change"].flatten()

    with open(posteriorsavepath, "rb") as handle:
        posterior_list = pkl.load(handle)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    color_idx_list = np.linspace(0, 1, len(np.unique(change_magnitude)))
    for color_idx, change_mag in zip(color_idx_list, np.unique(change_magnitude)):
        change_index = np.where(change_magnitude == change_mag)[0]
        windowed_posterior = list()
        for index in change_index:
            start_index = int(change_time[index]-1 - window_length/2) # change time uses 1 indexing
            end_index = int(change_time[index]-1 + window_length/2)
            posterior = posterior_list[index][start_index:end_index]
            windowed_posterior.append(posterior)

        windowed_posterior_array = np.array(windowed_posterior)
        windowed_posterior_array = numpy_fillna(windowed_posterior_array)
        mean_posterior = np.nanmean(windowed_posterior_array, axis=0)
        std_posterior = np.nanstd(windowed_posterior_array.astype(float), axis=0) # somehow need to convert to float
        # TODO: include the STD.

        ax.plot(np.arange(1, window_length+1) - window_length/2, mean_posterior, label=str(np.exp(change_mag)),
                color=plt.cm.cool(color_idx), lw=2)

        # TODO fill in the std as well
        mean_posterior = mean_posterior.astype(float)
        ax.fill_between(np.arange(1, window_length+1) - window_length/2, mean_posterior - std_posterior,
                        mean_posterior + std_posterior, color=plt.cm.cool(color_idx), alpha=0.3)


    ax.set_xlabel("Peri-change time (frames)")
    ax.set_ylabel(r"$P(z_k = \mathrm{change} \vert x_{1:k})$")
    ax.legend(title="Change magnitude", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([-0.05, 1.05])

    # TODO: Plot the individual posterior as well


    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


def get_threshold_behaviour(datapath, posteriorsavepath, df_path, threshold=0.5):

    with open(posteriorsavepath, "rb") as handle:
        posterior_list = pkl.load(handle)

    with open(datapath, "rb") as handle:
        exp_data = pkl.load(handle)

    change_magnitude = exp_data["sig"].flatten()
    change_time = exp_data["change"].flatten()

    # simply compares the rt with the actual change time to look at performance of the model acting directly from the
    # posterior with some threshold

    lick_time_list = list()
    for posterior in posterior_list:
        lick_time = np.where(posterior >= threshold)[0]
        if len(lick_time) != 0:
            lick_time_list.append(lick_time[0])
        else:
            lick_time_list.append(np.nan)


    model_correct_lick = (lick_time_list >= change_time-1).astype(float) # again, beware of the 0 indexing.

    model_df = dict()
    model_df["change_magnitude"] = change_magnitude
    model_df["change_time"] = change_time
    model_df["model_rt"] = lick_time_list
    model_df["peri_stimulus_rt"] = lick_time_list - (change_time -1) # beware of 0 indexing.
    model_df["correct_lick"] = model_correct_lick

    model_df = pd.DataFrame.from_dict(model_df)

    with open(df_path, "wb") as handle:
        pkl.dump(model_df, handle)

def plot_threshold_beahviour(df_path, figsavepath=None, metric="correct_lick"):

    model_df = pd.read_pickle(df_path)
    model_pivot = model_df.groupby(["change_magnitude"], as_index=False).agg({metric: "mean"})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(model_pivot["change_magnitude"], model_pivot[metric], linewidth=2)
    ax.scatter(model_pivot["change_magnitude"], model_pivot[metric], linewidth=2)
    ax.set_xlabel("Change magnitude")
    ax.set_ylabel(metric)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsavepath is not None:
        plt.savefig(figsavepath, dpi=300)

    plt.show()


def main(run_get_posterior=False, run_plot_posterior=False, run_get_threshold_behaviour=False):
    # datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/subsetted_data/data_IO_083.pkl"
    # model_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/model_response_083_16.pkl"
    # fig_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/figures/model_response_083_17.png"
    # training_savepath = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/training_result_083_17.pkl"
    # vec_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/model_response_083_17.pkl"
    datapath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/exp_data/subsetted_data/data_IO_083.pkl")
    model_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/model_response_083_20.pkl")
    fig_save_path = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/model_response_083_20.png")
    training_savepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/hmm_data/training_result_083_20.pkl")



    """
    param: 
    1. k parameter in the sigmoid function
    2. false_negative value in cost-benefit function
    3. midpoint of the sigmoid decision function
    """

    # test_forward_algorithm()
    # test_on_data(change_val=1.35, exp_data_file=datapath)
    # simple_gradient_descent(datapath, num_epoch=100, param_vals=np.array([9.59, 0.27, 0.37]),
    #                         training_savepath=training_savepath)

    # gradient_descent_fit_vector(datapath, training_savepath, init_param_vals=np.array([10.0, 0.0, 0.5]),
    #                               time_shift_list=np.arange(18, 22), num_epoch=50)

    # plot_training_result(training_savepath)
    # run_through_entire_dataset(datapath=datapath, savepath=model_save_path, param=[9.23, 0.67, 0.03], numtrial=None)
    # compare_model_with_behaviour(model_behaviour_df_path=model_save_path, savepath=fig_save_path, showfig=True)

    # run_through_dataset_fit_vector(datapath=datapath, savepath=model_save_path, param=[10.0, 0.0, 0.5])
    # plot_fit_vector_performance(model_save_path, fig_save_path)


    # figsavepath = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model/figures/test_alpha.png")
    # test_loss_function(datapath, plot_example=False, savepath=figsavepath)

    main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    # figsavepath = os.path.join(main_folder, "figures/pure_posterior_constant_hazard_rate_window_80.png")
    # posteriorsavepath = os.path.join(main_folder, "mouse_83_pure_posterior_constant_hazard_rate.pkl")

    additional_info = "change_state_dependent_mean_estimate_p05_std_estimate_0"
    # additional_info = "clean"
    figsavepath = os.path.join(main_folder, "figures/pure_posterior_exp_transition_matrix_window_40_" + additional_info + ".png")
    posteriorsavepath = os.path.join(main_folder, "mouse_83_pure_posterior_exp_transition_matrix_" + additional_info + ".pkl")


    if run_get_posterior is True:
        get_posterior(datapath, posteriorsavepath=posteriorsavepath, noisy=True)

    if run_plot_posterior is True:
        plot_posterior(datapath=datapath, posteriorsavepath=posteriorsavepath, figsavepath=figsavepath,
                     window_length=40)

    if run_get_threshold_behaviour is True:
        decision_threshold = 0.5
        df_path = os.path.join(main_folder, "mouse_83_pure_posterior_constant_hazard_rate_df.pkl")
        get_threshold_behaviour(datapath=datapath, posteriorsavepath=posteriorsavepath,
                                df_path=df_path, threshold=decision_threshold)

        figsavepath = os.path.join(main_folder, "figures/mouse_83_pure_posterior_constant_hazard_rate_correct_lick_" +
                                   str(decision_threshold) + ".png")
        plot_threshold_beahviour(df_path=df_path, metric="correct_lick")
        figsavepath = os.path.join(main_folder, "figures/mouse_83_pure_posterior_peri_stimulus_rt_" +
                                   str(decision_threshold) + ".png")
        plot_threshold_beahviour(df_path=df_path, metric="peri_stimulus_rt")


if __name__ == "__main__":
    main(run_get_posterior=True, run_plot_posterior=True, run_get_threshold_behaviour=False)

