import normative_plot as nmt_plot
import evaluate_model as ev_model
import matplotlib.pyplot as plt
# import hmm2_jax as hmmjax
import os
import pickle as pkl
import numpy as np

"""
time_shift_array=[[6, 6, 6, 6, 7, 7],
                [6, 7, 7, 8, 9, 8],
                [9, 8, 9, 9, 10, 10],
                [8, 6, 8, 8, 8, 8],
                [7, 6, 7, 8, 8, 8],
                [7, 6, 7, 7, 7, 7]]
"""


def run_model_comparison(model_data_folder, main_folder, model_number_list, mouse_number_list,
                         model_comparison_df_savepath,
                         time_shift_array):
    model_comparison_df = ev_model.eval_model_across_mice(model_df_folder_path=model_data_folder,
                                                          mouse_df_folder_path=main_folder,
                                                          model_number_list=model_number_list,
                                                          time_shift_array=time_shift_array
                                                          )
    with open(model_comparison_df_savepath, "wb") as handle:
        pkl.dump(model_comparison_df, handle)


def main(model_number_list=[63, 64], model_names=["model A", "model B"],
         time_shift_array=[[6, 6, 6, 6, 7, 7],
                           [6, 7, 7, 8, 9, 8],
                           [9, 8, 9, 9, 10, 10],
                           [8, 6, 8, 8, 8, 8],
                           [7, 6, 7, 8, 8, 8],
                           [7, 6, 7, 7, 7, 7]],
         mouse_number_list=[75, 78, 79, 80, 81, 83], filecheck=False):
    # Specify path of data
    # mouse_number = 83
    home = os.path.expanduser("~")

    # main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    main_folder = os.path.join("/media/timothysit/180C-2DDD/second_rotation_project/")
    model_data_folder = os.path.join(main_folder, "hmm_data")
    # mouse_df_path = os.path.join(main_folder, "mouse_" + str(mouse_number) + "_df.pkl")
    fig_folder_path = os.path.join(main_folder, "figures", "model_evaluation")
    if not os.path.exists(fig_folder_path):
        os.mkdir(fig_folder_path)

    # Do an initial check that all files required loads properly
    # Otherwise EOF error will take a long time to return
    if filecheck is True:
        for mouse_idx in np.arange(0, np.shape(np.array(time_shift_array))[0]):
            for model_idx in np.arange(0, np.shape(np.array(time_shift_array))[1]):
                sample_file_name = os.path.join(model_data_folder,
                                    "model_individual_samples_0" + str(mouse_number_list[mouse_idx]) + "_"
                                   + str(model_number_list[model_idx])
                                   + "_time_shift_" + str(np.array(time_shift_array)[mouse_idx, model_idx]) + ".pkl")

                try:
                    with open(sample_file_name, "rb") as handle:
                        sample_file = pkl.load(handle)
                except:
                    print("Cannot open file: ", sample_file_name)

        print("Finished checking that all files load properly.")


    # model comparison
    model_comparison_df_savepath = os.path.join(main_folder, "hmm_data", "model_comparison_" +
                                                "_".join(str(x) for x in model_number_list) + ".pkl")
    if not os.path.exists(model_comparison_df_savepath):  # only run if the file does not exist
        run_model_comparison(model_data_folder=model_data_folder, main_folder=main_folder,
                             model_number_list=model_number_list,
                             model_comparison_df_savepath=model_comparison_df_savepath,
                             mouse_number_list=mouse_number_list,
                             time_shift_array=time_shift_array)

    with open(model_comparison_df_savepath, "rb") as handle:
        model_comparison_df = pkl.load(handle)

    # reaction time performance
    rt_metric_list = ["FA_rt_dist_fit", "hit_rt_dist_fit_1", "hit_rt_dist_fit_1p25",
                      "hit_rt_dist_fit_1p35",
                     "hit_rt_dist_fit_1p5", "hit_rt_dist_fit_2", "hit_rt_dist_fit_4"]
    # the first figure is a dummy figure to handle the error from ts.mplstyle

    # change model number ordering based on model_number_list
    for model_number, new_model_number in zip(model_number_list, np.arange(0, len(model_number_list))):
        model_comparison_df = model_comparison_df.replace({"model_number": model_number}, new_model_number)

    for rt_metric in rt_metric_list:
        fig = nmt_plot.plot_multi_model_rt_error(model_comparison_df, multiple_mice=True,
                                                 model_names=model_names, rt_metric=rt_metric)
        figsavepath = os.path.join(fig_folder_path, rt_metric + "_" + "model_" +
                                   "_".join([str(x) for x in model_number_list]))
        fig.set_size_inches(6, 4)
        fig.savefig(figsavepath)

    # psychometric mse, validation loss, and chronometric mse
    file_name_list = ["psychometric_mse", "min_per_sample_val_loss", "hit_chronometric_fit"]
    metric_list = ["hit_exclude_FA_fit", "min_per_sample_val_loss", "hit_chronometric_fit"]
    for file_name, metric in zip(file_name_list, metric_list):
        figsavepath = os.path.join(fig_folder_path, file_name + "model_" +
                                       "_".join([str(x) for x in model_number_list]))
        fig, ax = nmt_plot.plot_multi_model_psychometric_error(model_comparison_df, multiple_mice=True,
                                            model_names=model_names,
                                            metric=metric)
        if metric == "min_per_sample_val_loss":
            ax.set_ylabel("Per trial validation loss")

        fig.set_size_inches(6, 4)
        fig.savefig(figsavepath)


if __name__ == "__main__":
    main(model_number_list=[68, 69, 67, 70, 71, 72],
         model_names=['1', '2', '3', '4', '5', '6'],
         time_shift_array=[[6, 6, 6, 6, 7, 7],
                           [6, 7, 7, 8, 9, 8],
                           [9, 8, 9, 9, 10, 10],
                           [8, 6, 8, 8, 8, 8],
                           [7, 6, 7, 8, 8, 8],
                           [7, 6, 7, 7, 7, 7]],
         mouse_number_list=[75, 78, 79, 80, 81, 83],
         filecheck=False,
         )
    # model_names=["Constant \n hazard rate", "Experiment \n hazard rate", "Varying \n hazard rate",
    #              "Backward transition"]
