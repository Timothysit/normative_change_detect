# Automatic Differentiation using jax
# https://github.com/google/jax
import jax.numpy as np
import numpy as onp # original numpy for indexed assignment/mutation (outside context of differentiation)
from jax import grad, jit, vmap
from jax.experimental import optimizers

# Other things
import hmm  # functions to simulate data
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import itertools

import os
import pickle as pkl
import pandas as pd


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

def cal_p_x_given_zchange(x_k, z_n):
    zchange_mu_list = np.log(np.array([1.25, 1.35, 1.50, 2.00, 4.00]))
    zchange_var_list = np.array([0.25, 0.25, 0.25, 0.25, 0.25])

    zchange_mu = zchange_mu_list[z_n]
    zchange_var = zchange_var_list[z_n]

    p_x_given_zchange = (1 / np.sqrt(2 * np.pi * zchange_var)) * np.exp(-(x_k - zchange_mu) ** 2 / (2 * zchange_var))

    # returns row vector
    return p_x_given_zchange.T



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


    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_xk_given_z = cal_p_x_given_z(x_k=x[k])


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


def apply_strategy(prob_lick, k, midpoint=0.5):
    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    # two param logistic function
    max_val = 1  # L
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

    loss_val_matrix = np.zeros((num_epoch, len(time_shift_list)))
    final_param_list = list() # list of list, each list is the parameter values for a particular time shift



    for time_shift in time_shift_list:

        step_size = 0.001
        momentum_mass = 0.9
        opt_init, opt_update = optimizers.momentum(step_size, mass=momentum_mass)

        @jit
        def update(i, opt_state):
            params = optimizers.get_params(opt_state)
            return opt_update(i, grad(loss_function_fit_vector)(params, None), opt_state)

        opt_state = opt_init(init_param_vals)

        # TODO: think about doing batch gradient descent
        itercount = itertools.count()
        for epoch in range(num_epoch):
            opt_state = update(next(itercount), opt_state)
            params = optimizers.get_params(opt_state)
            loss_val = loss_function_fit_vector(params)
            print("Loss:", loss_val)




        final_param_list.append(trained_params)

    training_result = dict()
    training_result["loss"] = loss_val_matrix
    training_result["param_val"] = final_param_list

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


def main():
    datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/subsetted_data/data_IO_083.pkl"
    model_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/model_response_083_14.pkl"
    fig_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/figures/model_response_083_14.png"
    training_savepath = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/training_result_083_18.pkl"

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

    gradient_descent_fit_vector(exp_data_path=datapath,
                                training_savepath=training_savepath,
                                init_param_vals=np.array([10.0, 0.0, 0.5]),
                                time_shift_list=np.arange(0, 5), num_epoch=10)


if __name__ == "__main__":
    main()

