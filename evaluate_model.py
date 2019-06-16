# Functions to evaluate the performance of models in terms of fitting behaviour data

# Loading things
import os
import pandas as pd

# Evaluation metrics
import numpy as np
# import sklearn as skl
import scipy.stats as spstats
import sklearn.metrics

# Storing things
from collections import defaultdict  # intialise dict of lists

from tqdm import tqdm # Loading bar

def compare_psychometric_fit(mouse_psychometric, model_psychometric, method="mse"):
    """

    :param mouse_psychometric:
    :param model_psychometric:
    :param method: method used to quantify the difference between the mouse and mode psychometric curve
        "mse" | mean-square error of the two curves
    :return:
    """

    if method == "mse":
        # mse = skl.metrics.mean_squared_error(mouse_psychometric, model_psychometric)
        mse = sklearn.metrics.mean_squared_error(mouse_psychometric, model_psychometric)
    return mse


def compare_dist_fit(mouse_dist, model_dist, method="ks_continuous"):

    # TODO: May have to implement discrete version (or borrow it from R)

    if method == "ks_continuous":
        ks_stat, ks_p_value = spstats.ks_2samp(mouse_dist, model_dist)

    return ks_stat, ks_p_value


def get_psychometric_data(mouse_df, model_df, psychometric_type="hit_exclude_FA"):

    if psychometric_type == "hit_exclude_FA":
        metric = "lick"
        mouse_metric = "mouse_lick"

        # subset model df based on model FA
        model_no_FA_index = np.where(model_df["FA"] != 1)[0]
        model_df = model_df.iloc[model_no_FA_index]

        # subset model df based on mouse FA
        mouse_no_FA_index = np.where(mouse_df["mouse_FA"] != 1)[0]
        model_df = model_df.loc[model_df["trial_ID"].isin(mouse_no_FA_index)]


        mouse_no_FA_index = np.where(mouse_df["mouse_FA"] != 1)[0]
        mouse_df = mouse_df.iloc[mouse_no_FA_index]


        model_df_all_simulations = model_df.groupby(["change", "sample_ID"], as_index=False).agg({metric: "mean"})
        model_df_mean = model_df_all_simulations.groupby(["change"], as_index=False).agg({metric: "mean"})
        model_df_sem = model_df_all_simulations.groupby(["change"], as_index=False).agg({metric: "sem"})

        mouse_df_mean = mouse_df.groupby(["change"], as_index=False).agg({mouse_metric: "mean"})


        model_and_mouse_psychometric_df = pd.DataFrame({
            "mouse_mean": mouse_df_mean[mouse_metric],
            "model_mean": model_df_mean[metric],
            "model_error": model_df_sem[metric],
            "change": np.exp(model_df_mean["change"])
        })

    elif psychometric_type == "hit_chronometric":
        model_metric = "peri_stimulus_rt"
        mouse_metric = "peri_stimulus_rt"
        summary_stat = "median"

        model_hit_index = np.where(model_df["correct_lick"] == 1)[0]
        mouse_hit_index = np.where(mouse_df["mouse_hit"] == 1)[0]

        model_df = model_df.iloc[model_hit_index]
        mouse_df = mouse_df.iloc[mouse_hit_index]

        model_pivot_df = model_df.groupby(["change"], as_index=False).agg({model_metric: summary_stat})
        model_df_error = model_df.groupby(["change"], as_index=False).agg({model_metric: "sem"})
        mouse_pivot_df = mouse_df.groupby(["change"], as_index=False).agg({mouse_metric: summary_stat})


        model_and_mouse_psychometric_df = pd.DataFrame({
            "mouse_median": mouse_pivot_df[mouse_metric],
            "model_median": model_pivot_df[model_metric],
            "model_error": model_df_error[model_metric],
            "change": np.exp(model_pivot_df["change"])
        })


    return model_and_mouse_psychometric_df


def get_reaction_time_data(mouse_df, model_df, outcome="FA", rt_type="absolute_decision_time",
                           model_sub_sample=None, change_magnitude=None):

    if outcome == "FA":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 & (mouse_df["change"] > 0))[0]
        mouse_df = mouse_df.loc[~mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 & (model_df["change"] > 0))[0]
        model_df = model_df.loc[~model_df.index.isin(model_hit_index)]
    elif outcome == "Hit" and (change_magnitude is None):
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 & (np.exp(mouse_df["change"]) > 1.0))[0]
        mouse_df = mouse_df.loc[mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 & (np.exp(model_df["change"]) > 1.0))[0]
        model_df = model_df.loc[model_df.index.isin(model_hit_index)]
    elif outcome == "Hit":
        mouse_hit_index = np.where((mouse_df["peri_stimulus_rt"]) >= 0 &
                                   (np.exp(mouse_df["change"]) == change_magnitude))[0]
        mouse_df = mouse_df.loc[mouse_df.index.isin(mouse_hit_index)]
        model_hit_index = np.where((model_df["peri_stimulus_rt"]) >= 0 &
                                   (np.exp(model_df["change"]) == change_magnitude))[0]
        model_df = model_df.loc[model_df.index.isin(model_hit_index)]


    if model_sub_sample is not None:
        model_df = model_df.sample(model_sub_sample)

    mouse_rt = mouse_df[rt_type]
    model_rt = model_df[rt_type]

    return mouse_rt, model_rt



def eval_model(mouse_df_path, model_df_folder_path, model_number_list=[36, 100], mouse_number=83,
               rt_sub_sample=None, time_shift_list=None):
    """

    :param mouse_df_path:
    :param model_df_folder_path:
    :param model_number_list:
    :param mouse_number:
    :param rt_sub_sample:
    :param time_shift_fitted: whether the time shift is fitted in the model data frames
    :param custom_time_shift_list: custom time shift values to use when computing the cost and mse. If None, use
    the time shift that minimises cross-validation loss.
    :return:
    """

    mouse_df = pd.read_pickle(mouse_df_path)
    model_df_file_list = os.listdir(model_df_folder_path)

    min_training_loss_list = list()
    min_val_loss_list = list()
    min_per_sample_train_loss_list = list()
    min_per_sample_val_loss_list = list()
    hit_exclude_FA_fit_score_list = list()
    hit_chronometric_fit_list = list()
    FA_rt_dist_score_list = list()
    hit_rt_dist_score_dict = dict()
    # TODO: compute median reaction time
    psychometric_df_list = list()
    chronometric_df_list = list()

    rt_output_dict = defaultdict(list)

    change_magnitude_list = [1.00, 1.25, 1.35, 1.5, 2.0, 4.0]
    for change_magnitude in change_magnitude_list:
        hit_rt_dist_score_dict[change_magnitude] = list()


    for model_number, time_shift in zip(model_number_list, time_shift_list):
        # get model loss and parameters
        model_file_name = "training_result_0" + str(mouse_number) + "_" + str(model_number) + ".pkl"
        model_training_result = pd.read_pickle(os.path.join(model_df_folder_path, model_file_name))
        min_validation_loss = np.min(model_training_result["val_loss"])
        min_val_loss_index = np.where(np.array(model_training_result["val_loss"] == min_validation_loss))[0][0]
        if np.array(model_training_result["train_loss"]).ndim == 0:  # 0-dimension array in control models
            min_training_loss = model_training_result["train_loss"].item()  # convert 0d array to scalar
        else:
            min_training_loss = model_training_result["train_loss"][min_val_loss_index]

        # TODO: test loss still needs work (from hmm2_jax.py)
        # test_loss = model_training_result["test_loss"]  # no need to min

        # TODO: Remove once all data is moved to the new format with this information already
        min_per_sample_train_loss = min_training_loss / model_training_result["train_set_size"]
        min_per_sample_val_loss = min_validation_loss / model_training_result["val_set_size"]

        # TODO Something strange here for model 65 (will need to double check)
        # TODO: need to get test loss saved in hmm2_jax.py
        # if type(test_loss) is list:
        #     per_sample_test_loss = test_loss[0] / model_training_result["test_set_size"]
        # else:
        #     per_sample_test_loss = test_loss.item() / model_training_result["test_set_size"]

        min_training_loss_list.append(min_training_loss)
        min_val_loss_list.append(min_validation_loss)

        min_per_sample_train_loss_list.append(min_per_sample_train_loss)
        min_per_sample_val_loss_list.append(min_per_sample_val_loss)

        # get simulated results from model and compare with mouse data
        simulated_results_file = "model_individual_samples_0" + str(mouse_number) + "_" + str(model_number) + \
                                 "_time_shift_" + str(time_shift) + ".pkl"
        model_df = pd.read_pickle(os.path.join(model_df_folder_path, simulated_results_file))


        # Calculate error on fitting psychometric curve
        model_and_mouse_psychometric_df = get_psychometric_data(mouse_df, model_df,
                                                                psychometric_type="hit_exclude_FA")

        # add mouse number, model number, and time shift to the df (to be used for plotting)
        model_and_mouse_psychometric_df["model_number"] = np.repeat(model_number, len(model_and_mouse_psychometric_df))
        model_and_mouse_psychometric_df["mouse_number"] = np.repeat(mouse_number, len(model_and_mouse_psychometric_df))
        model_and_mouse_psychometric_df["time_shift"] = np.repeat(time_shift, len(model_and_mouse_psychometric_df))
        psychometric_df_list.append(model_and_mouse_psychometric_df)

        hit_exclude_FA_fit_score = compare_psychometric_fit(
                                    mouse_psychometric=model_and_mouse_psychometric_df["mouse_mean"],
                                    model_psychometric=model_and_mouse_psychometric_df["model_mean"])
        hit_exclude_FA_fit_score_list.append(hit_exclude_FA_fit_score)

        # Calculate error on fitting chronometric curve
        model_and_mouse_psychometric_df = get_psychometric_data(mouse_df, model_df,
                                                                psychometric_type="hit_chronometric")
        hit_chronometric_fit_score = compare_psychometric_fit(
                                    mouse_psychometric=model_and_mouse_psychometric_df["mouse_median"],
                                    model_psychometric=model_and_mouse_psychometric_df["model_median"])
        hit_chronometric_fit_list.append(hit_chronometric_fit_score)

        model_and_mouse_psychometric_df["model_number"] = np.repeat(model_number, len(model_and_mouse_psychometric_df))
        model_and_mouse_psychometric_df["mouse_number"] = np.repeat(mouse_number, len(model_and_mouse_psychometric_df))
        model_and_mouse_psychometric_df["time_shift"] = np.repeat(time_shift, len(model_and_mouse_psychometric_df))
        chronometric_df_list.append(model_and_mouse_psychometric_df)


        # Calculate error on fitting reaction time
        mouse_rt_FA, model_rt_FA = get_reaction_time_data(mouse_df, model_df, outcome="FA",
                                                          rt_type="absolute_decision_time",
                           model_sub_sample=len(mouse_df))

        # TODO: still need to initalise dictionary and make the entry lists.
        rt_output_dict["mouse_number"].append(mouse_number)
        rt_output_dict["model_number"].append(model_number)
        rt_output_dict["time_shift"].append(time_shift)
        rt_output_dict["mouse_rt_FA"].append(list(mouse_rt_FA))
        rt_output_dict["model_rt_FA"].append(list(model_rt_FA))

        ks_stat_FA, ks_p_value_FA = compare_dist_fit(mouse_rt_FA, model_rt_FA, method="ks_continuous")

        FA_rt_dist_score_list.append(ks_stat_FA)


        for change_magnitude in change_magnitude_list:
            mouse_rt, model_rt = get_reaction_time_data(mouse_df, model_df, outcome="Hit", rt_type="absolute_decision_time",
                                   model_sub_sample=len(mouse_df), change_magnitude=change_magnitude)
            ks_stat_hit, ks_p_value_hit = compare_dist_fit(mouse_rt, model_rt, method="ks_continuous")
            hit_rt_dist_score_dict[change_magnitude].append(ks_stat_hit)


    # Information required
    # Fitted parameters

    model_comparison_df = pd.DataFrame({"model_number": model_number_list,
                                        "time_shift": time_shift_list,
                                        "mouse_number": np.repeat(mouse_number, len(model_number_list)),
                                        "min_train_loss": min_training_loss_list,
                                        "min_val_loss": min_val_loss_list,
                                        "min_per_sample_train_loss": min_per_sample_train_loss_list,
                                        "min_per_sample_val_loss": min_per_sample_val_loss_list,
                                        # "per_sample_test_loss": per_sample_test_loss,
                                        "hit_exclude_FA_fit": hit_exclude_FA_fit_score_list,
                                        "hit_chronometric_fit": hit_chronometric_fit_list,
                                        "FA_rt_dist_fit": FA_rt_dist_score_list,
                                        "hit_rt_dist_fit_1": hit_rt_dist_score_dict[1.00],
                                        "hit_rt_dist_fit_1p25": hit_rt_dist_score_dict[1.25],
                                        "hit_rt_dist_fit_1p35": hit_rt_dist_score_dict[1.35],
                                        "hit_rt_dist_fit_1p5": hit_rt_dist_score_dict[1.35],
                                        "hit_rt_dist_fit_2": hit_rt_dist_score_dict[2.00],
                                        "hit_rt_dist_fit_4": hit_rt_dist_score_dict[4.00]
                                        })

    multiple_model_output_dict = {"psychometric_df_list": psychometric_df_list,
                                  "chronometric_df_list": chronometric_df_list,
                                  "rt_output_dict": rt_output_dict
                                  }

    return model_comparison_df, multiple_model_output_dict



def eval_model_across_mice(model_df_folder_path, mouse_df_folder_path,
                           mouse_number_list=[75, 78, 79, 80, 81, 83],
                           model_number_list=[68, 67],
                           time_shift_array=[[6, 6], [6, 7], [9, 9], [8, 8], [7, 7], [7, 7]]):
    """
    Takes in list of models, and return model comparison statistics.
    (Normally called by normative_hmm_main.py)
    :param model_df_folder_path:
    :param mouse_df_folder_path:
    :param mouse_number_list:
    :param model_number_list:
    :param time_shift_array: list of lists. each list is the time shift to use for a mouse in the models
    Therefore, the overall list will have the same length as the mouse number, and each sub-list will have the
    same length as the model number.
    :return:
    """

    model_comparison_df_list = list()
    multiple_model_output_dict_store = dict()
    for mouse_number, time_shift_list in tqdm(zip(mouse_number_list, time_shift_array)):
        mouse_df_path = os.path.join(mouse_df_folder_path, "mouse_" + str(mouse_number) + "_df.pkl")
        # print(mouse_df_path)
        model_comparison_df, multiple_model_output_dict = eval_model(mouse_df_path, model_df_folder_path,
                                         model_number_list=model_number_list, mouse_number=mouse_number,
                                         time_shift_list=time_shift_list)
        model_comparison_df_list.append(model_comparison_df)
        multiple_model_output_dict_store["mouse_" + str(mouse_number)] = multiple_model_output_dict

    mouse_model_comparison_df = pd.concat(model_comparison_df_list)


    return mouse_model_comparison_df, multiple_model_output_dict_store