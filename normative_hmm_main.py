import normative_plot as nmt_plot
import evaluate_model as ev_model
import matplotlib.pyplot as plt
# import hmm2_jax as hmmjax
import os
import pickle as pkl


def run_model_comparison(model_data_folder, main_folder, model_number_list, model_comparison_df_savepath):
    model_comparison_df = ev_model.eval_model_across_mice(model_df_folder_path=model_data_folder,
                                                          mouse_df_folder_path=main_folder,
                                                          mouse_number_list=[75, 78, 79, 80, 81, 83],
                                                          model_number_list=model_number_list,
                                                          )
    with open(model_comparison_df_savepath, "wb") as handle:
        pkl.dump(model_comparison_df, handle)


def main(model_number_list=[63, 64]):
    # Specify path of data
    mouse_number = 83
    home = os.path.expanduser("~")
    main_folder = os.path.join(home, "Dropbox/notes/Projects/second_rotation_project/normative_model")
    model_data_folder = os.path.join(main_folder, "hmm_data")
    mouse_df_path = os.path.join(main_folder, "mouse_" + str(mouse_number) + "_df.pkl")
    fig_folder_path = os.path.join(main_folder, "figures", "model_evaluation")
    if not os.path.exists(fig_folder_path):
        os.mkdir(fig_folder_path)


    # model comparison
    model_comparison_df_savepath = os.path.join(main_folder, "hmm_data", "model_comparison_" +
                                                "_".join(str(x) for x in model_number_list) + ".pkl")
    if not os.path.exists(model_comparison_df_savepath):  # only run if the file does not exist
        run_model_comparison(model_data_folder, main_folder, model_number_list, model_comparison_df_savepath)

    with open(model_comparison_df_savepath, "rb") as handle:
        model_comparison_df = pkl.load(handle)

    # reaction time performance
    rt_metric_list = ["FA_rt_dist_fit", "hit_rt_dist_fit_1", "hit_rt_dist_fit_1p25",
                      "hit_rt_dist_fit_1p35",
                     "hit_rt_dist_fit_1p5", "hit_rt_dist_fit_2", "hit_rt_dist_fit_4"]
    # the first figure is a dummy figure to handle the error from ts.mplstyle

    for rt_metric in rt_metric_list:
        fig = nmt_plot.plot_multi_model_rt_error(model_comparison_df, multiple_mice=True,
                                                 model_names=["Control", "Sigmoid decision"], rt_metric=rt_metric)
        figsavepath = os.path.join(fig_folder_path, rt_metric + "_" + "model_" +
                                   "_".join([str(x) for x in model_number_list]))
        fig.set_size_inches(4, 4)
        fig.savefig(figsavepath)

    # psychometric curve performance
    file_name_list = ["psychometric_mse", "min_per_sample_val_loss"]
    metric_list = ["hit_exclude_FA_fit", "min_per_sample_val_loss"]
    for file_name, metric in zip(file_name_list, metric_list):
        figsavepath = os.path.join(fig_folder_path, file_name + "model_" +
                                       "_".join([str(x) for x in model_number_list]))
        fig, ax = nmt_plot.plot_multi_model_psychometric_error(model_comparison_df, multiple_mice=True,
                                            model_names=["Control", "Sigmoid decision"],
                                            metric=metric)
        if metric == "min_per_sample_val_loss":
            ax.set_ylabel("Per sample validation loss")

        fig.set_size_inches(4, 4)
        fig.savefig(figsavepath)

    # chronometric curve comparison TODO

    # Validation loss



if __name__ == "__main__":
    main(model_number_list=[63, 64])