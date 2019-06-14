import pickle as pkl
import pandas as pd
import scipy.io
import glob
import os
import numpy as np


"""
Script to process exp data

"""


from os.path import expanduser

def main():
    home = expanduser("~")
    mainfolder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")


    exp_data_folder = os.path.join(mainfolder, "exp_data/data/")
    processed_data_folder = os.path.join(mainfolder, "exp_data/subsetted_data/")

    exp_file_names = [f for f in glob.glob(exp_data_folder + "**/*.mat", recursive=True)]

    for exp_file_name in exp_file_names:
        exp_data = scipy.io.loadmat(exp_file_name)

        noiseless_trial_type = exp_data["noiseless"].flatten()
        mouse_abort = (exp_data["outcome"].flatten() == "abort").astype(float)
        nonsplit_block = (exp_data["hazard"].flatten() == "nonsplit").astype(float)
        early_block = (exp_data["hazard"].flatten() == "early").astype(float)
        late_block = (exp_data["hazard"].flatten() == "late").astype(float)

        # remove running trials
        mouse_move = (exp_data["running"] == "running").flatten().astype(float)
        # subset_trial_index = np.where( (noiseless_trial_type == 0) & (mouse_abort == 0) & (mouse_move == 0) )[0]

        # Subset criteria
        subset_trial_index = np.where((noiseless_trial_type == 0) & (mouse_abort == 0) & (early_block == 1))[0]


        # allow running trials
        # subset_trial_index = np.where((noiseless_trial_type == 0) & (mouse_abort == 0))[0]

        # subset based on block type (nonsplit, early, late)

        new_exp_data = dict()
        keys_of_interest = ["ys", "rt", "change", "sig", "sig_avg", "sig_std", "noiseless", "hazard", "running", "outcome"]
        exp_data["ys"] = exp_data["ys"].T
        exp_data["rt"] = exp_data["rt"].T

        # TODO: include info about the subset crieteria in the file.
        exp_data["subset_criteria"] = ["noisy", "no_abort", "early_block"]


        for key_to_inquire in keys_of_interest:
            new_exp_data[key_to_inquire] = exp_data[key_to_inquire][subset_trial_index]

        # save new data in processed_data_folder
        new_file_name = os.path.basename(exp_file_name + "_early_blocks")
        new_file_name = new_file_name.replace(".mat", ".pkl")

        with open(os.path.join(processed_data_folder, new_file_name), "wb") as handle:
            pkl.dump(new_exp_data, handle)


if __name__ == "__main__":
    main()

