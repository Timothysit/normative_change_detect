import autograd.numpy as np
from autograd import grad
import hmm # functions to simulate data
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm

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
    p_x_given_z2 = np.dot(p_x_given_z2, z2_weights)

    return p_x_given_z2

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
    transition_matrix = np.array([[1 - hazard_rate, hazard_rate], [0, 1]])
    init_state_probability = np.array([1.0, 0.0])

    # Iniitial probabilities
    p_z1_and_x = cal_p_x_given_z1(x_k=x[0]) * init_state_probability[0]
    p_z2_and_x = cal_p_x_given_z2(x_k=x[0]) * init_state_probability[1]

    # Loop through the rest of the samples to compute P(x_k, z_k) for each
    for k in np.arange(1, len(x)):
        p_z1_and_x = cal_p_x_given_z1(x_k=x[k]) * transition_matrix[0, 0] * p_z1_and_x + \
                          cal_p_x_given_z2(x_k=x[k]) * transition_matrix[1, 0] * p_z2_and_x
        p_z2_and_x = cal_p_x_given_z1(x_k=x[k]) * transition_matrix[0, 1] * p_z1_and_x + \
                           cal_p_x_given_z2(x_k=x[k]) * transition_matrix[1, 1] * p_z2_and_x
        # p_z_and_x[k, 1] = 1 - p_z_and_x[k, 0]

        # NOTE: This step is not the conventional forward algorithm, but works.
        p_x = p_z1_and_x + p_z2_and_x
        p_z1_given_x.append(p_z1_and_x / p_x) # p(x_{1:k})
        p_z2_given_x.append(p_z2_and_x / p_x)

    p_z_given_x = np.column_stack((p_z1_given_x, p_z2_given_x))

    return p_z_given_x



def apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=1.0, false_negative=1.0, false_positive=1.0):
    """
    computes decision value based on cost and benefit
    note that true-positive is always set to 1.0, everything else is relative to it
    :param true_positive:
    :return:
    """

    lick_benefit = change_posterior * true_positive
    lick_cost = change_posterior * false_positive
    no_lick_benefit = (1 - change_posterior) * true_negative
    no_lick_cost = (1 - change_posterior) * false_negative

    prob_lick = (lick_benefit + no_lick_cost - lick_cost - no_lick_benefit) / (true_positive +
                 true_negative + false_negative + false_positive)

    return prob_lick


def apply_strategy(prob_lick, k):

    # four param logistic function
    # prob_lick = y = a * 1 / (1 + np.exp(-k * (x - x0))) + b

    # two param logistic function
    max_val = 1 # L
    midpoint = 0.5 # x0
    prob_lick = max_val / (1 + np.exp(-k * (prob_lick - midpoint)))

    return prob_lick

def loss_function(param_vals):
    # missing arguments: signals, param_vals_, actual_lick_,
    """
    loss between predicted licks and actual licks
    computed using cross entropy
    This loss function is for licking versus no-licking, but does not compute the loss of the reaction time.
    :return:
    """

    cross_entropy_loss = 0
    epsilon = 0.001  # small value to avoid taking the log of 0
    for x_num, x in enumerate(signals):
        posterior = forward_inference(x.flatten())
        p_lick = apply_cost_benefit(change_posterior=posterior[:, 1], true_negative=1.0, false_negative=1.0, false_positive=1.0)
        p_lick = apply_strategy(p_lick, k=param_vals[0])
        p_lick = np.max(p_lick)
        cross_entropy_loss += -(actual_lick[x_num] * np.log(p_lick + epsilon) + (1 - actual_lick[x_num]) * np.log(
            1 - p_lick + epsilon))

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

def test_on_data(change_val=1.0):
    exp_data_file_list = ["/media/timothysit/180C-2DDD/second_rotation_project/exp_data/data/data_IO_083.mat"]
    for exp_data_file in exp_data_file_list:
        exp_data = scipy.io.loadmat(exp_data_file)

    trial_type_list = exp_data["hazard"]
    change_magnitude = np.exp(exp_data["sig"].flatten())
    noiseless_trial_type = exp_data["noiseless"].flatten()
    mouse_abort = (exp_data["outcome"].flatten() == "abort").astype(float)

    # remove aborted trials, and noisless trials
    if change_magnitude is not None:
        trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0) * (change_magnitude == change_val))[0]
    else:
        trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0))[0]

    signal = exp_data["ys"].flatten()[trial_index][0][0]
    tau = exp_data["change"][trial_index][0][0]

    p_z_given_x = forward_inference(signal)

    # Plot example
    plot_signal_and_inference(signal=signal, tau=tau, prob=p_z_given_x[:, 1])

    return None


def posterior_to_decision(posterior, return_prob=True):
    change_posterior = posterior[:, 1] # p(z_2 | x_{1:k})
    p_lick = apply_cost_benefit(change_posterior, true_positive=1.0, true_negative=1.0,
                                false_negative=1.0, false_positive=1.0)
    p_lick = apply_strategy(p_lick, k=1.0)

    if return_prob is True:
        return np.max(p_lick)


def run_through_entire_dataset(datapath, savepath, subset_criteria=["abort"], numtrial=100):
    # note this will also include experimental data, so perhaps "model_behaviour" is not the best variable name...
    # runs forward algorithm through entire dataset
    exp_data = scipy.io.loadmat(datapath)

    signals = exp_data["ys"].flatten()
    change = np.exp(exp_data["sig"].flatten())
    tau = exp_data["change"].flatten()

    # Experimental data
    mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()

    if "abort" in subset_criteria:
        no_abort_index = np.where(exp_data["outcome"].flatten() != "abort")[0]
        signals = signals[no_abort_index]
        change = change[no_abort_index]
        tau = tau[no_abort_index]

    decision = list()
    decision_time = list()
    lick_choice_list = list()

    for signal in tqdm(signals[0:numtrial]):
        signal = signal.reshape(-1, 1)

        posterior = forward_inference(signal)

        lick_choice = posterior_to_decision(posterior, return_prob=True)

        # save data
        decision.append(np.max(lick_choice)) # 0 = no lick, 1 = lick

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
    model_behaviour["decision"] = decision
    model_behaviour["tau"] = tau[0:numtrial]
    model_behaviour["mouse_lick"] = mouse_lick[0:numtrial]

    # model_behaviour["lick_choice_list"] = lick_choice_list

    # Convert from dictionary to dataframe
    model_behaviour_df = pd.DataFrame.from_dict(model_behaviour)


    with open(savepath, "wb") as handle:
        pkl.dump(model_behaviour_df, handle)


def simple_gradient_descent(num_epoch=100):
    # Define targets (actual licks)
    exp_data_file = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/data/data_IO_083.mat"
    exp_data = scipy.io.loadmat(exp_data_file)


    reaction_time = exp_data["rt"]
    mouse_hit = (exp_data["outcome"] == "Hit").astype(float).flatten()
    mouse_FA = (exp_data["outcome"] == "FA").astype(float).flatten()
    mouse_lick = np.any([mouse_hit, mouse_FA], axis=0).astype(float).flatten()

    signal = exp_data["ys"].flatten()

    # subset trials
    noiseless_trial_type = exp_data["noiseless"].flatten()
    mouse_abort = (exp_data["outcome"].flatten() == "abort").astype(float)
    trial_index = np.where((mouse_abort == 0) & (noiseless_trial_type == 0))[0]
    mouse_lick = mouse_lick[trial_index]
    signal = signal[trial_index]

    # only get first 100 trials just as a test
    mouse_lick = mouse_lick[0:100]
    signal = signal[0:100]


    # Simple Gradient Descent
    loss_function_grad = grad(loss_function)
    param_vals = np.array([1.0])

    # Define global variables used by loss_function
    global actual_lick
    global signals
    actual_lick = mouse_lick
    signals = signal


    # loss_value = loss_function(actual_lick=mouse_lick, signals=signal, param_vals=param_vals)

    learning_rate = 0.01

    print("Initial loss:", loss_function_grad(param_vals))

    loss_val_list = list()
    num_param = 1
    param_val_array = np.zeros((num_epoch, num_param))
    for i in tqdm(range(num_epoch)):
        loss_val_list.append(loss_function_grad(param_vals))
        param_val_array[i, :] = param_vals
        param_vals -= loss_function_grad(param_vals) * learning_rate

    print("Trained loss:", loss_function_grad(param_vals))

    training_result = dict()
    training_result["loss"] = loss_val_list
    training_result["param_val"] = param_val_array

    training_savepath = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/training_result.pkl"

    with open(training_savepath, "wb") as handle:
        pkl.dump(training_result, handle)


def plot_training_result(training_savepath):

    with open(training_savepath, "rb") as handle:
        training_result = pkl.load(handle)

    loss = np.concatenate(training_result["loss"])
    param_val = np.concatenate(training_result["param_val"])

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(np.arange(1, len(loss)+1), -loss)
    axs[1].plot(np.arange(1, len(loss)+1), param_val)

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
    datapath = "/media/timothysit/180C-2DDD/second_rotation_project/exp_data/data/data_IO_083.mat"
    training_savepath = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/training_result.pkl"
    model_save_path = "/media/timothysit/180C-2DDD/second_rotation_project/hmm_data/model_response_083_4.pkl"

    # test_forward_algorithm()
    # test_on_data(change_val=2.0)
    # simple_gradient_descent()
    # plot_training_result(training_savepath)
    run_through_entire_dataset(datapath=datapath, subset_criteria=["abort"], savepath=model_save_path, numtrial=100)
    compare_model_with_behaviour(model_behaviour_df_path=model_save_path, savepath=None, showfig=True)

if __name__ == "__main__":
    main()

