# Normative models of change detection.

## Project description

We start with ideal observer models of change detection, and add biological constraints to model mice behaviour.

The ideal observer model: 

 - Hidden Markv Model with $M$ states, which includes 1 baseline state and $M-1$ change states 
 - Transition probabilities are either fixed as constant, derived from the experimental data, or fitted to the behaviourla data
 
Constraints / decision processes added to the model:


## Installation (TODO)

The installation was tested on devices with NVIDIA GPU running Ubuntu 18.04.1 LTS x86_64.

Basic dependencies (not requiring GPU-driver) can be installed in a conda environment using the YAML file: 

To install `JAX`, which is used to run gradient descent, please follow the instructions on their page [here](https://github.com/google/jax)


## Instructions 

The analysis is separated into 3 main components: 

 1. Model fitting
 2. Sampling from the model 
 3. Evaluating match of samples with behavioural data
 
 Where each step has its associated plots created using functions in `normative_plot.py`
 
 Model fitting is performed by running the script: `hmm2_jax.py`
 Sample from the model by running the script: `sample_from_model.py`
 Evaluate the model samples by running the script: `evaluate_model.py`
 
 Folder structure: 
 
  - currently, you have to change the `mainfolder` variable to the path where the experimental data and model data will live in 
  - the experiment-related matlab files should be in: `mainfolder/exp_data/data/`
  - the experiment-related pickle files will then be saved in `mainfolder/exp_data/subsetted_data/` (for now, you have to create these folders, they won't be created automatically)
   - all the files relating to the model will be saved in `mainfolder/hmm_data`
   - figures will be saved in `mainfolder/figures/`
 
 Converting from matlab files to pickle files: 
 
  - use `process_exp_data.py`
  - this is also where the data will be subsetted 
 
 
 Running `hmm2_jax.py`
 
  - specify which mouse/mice to run in `exp_data_number_list` 
  - specify the model number which you want to save to 
  - run `main` with the following options set to True (all other options set to False): 
     - `run_gradient_descent`
  - the latest function to run gradient descent is `gradient_descent_w_cv`
	 - specify the initial (non-hazard rate) parameters here 
	 - specify the number of non-hazard rate parameters in `n_params`
	 - specify list of time shift to run gradient descent on in `time_shift_list`

After runnning gradient descent, the data will be saved in `training_save_path`, and will contain, for each $n$ epoch, the parameter values, the training and cross validation loss, and the final test loss.


Plotting results of gradient descent: 

 - set `run_plot_training_loss` to True to plot the training and cross validation loss over epochs 
 - set `run_plot_trained_hazard_rate` to True to plot the hazard rate values in the epoch with the lowest cross validation loss (or a user-specified epoch number)

Preparing the model for sampling: 

 - set `run_model ` to True 
    - currently, the time shift has to be manually specified within the `run_through_dataset_fit_vector` function
	- this will read the training results, find the parameters with the lowest cross-validation loss, and obtain the p(lick) from the model over all the signals in the mouse data file 
	- output will be saved to `model_save_path`
	
	
Sampling from the model:

 - go to `sample_from_model.py`
 - specify the mouse and model number
 - if this is the first time a certain mouse is analyse, then set `run_get_mouse_df` to True to obtain the behavioural data for the mouse, the data will be saved in `mouse_behaviour_path`
 - set `run_sample_from_model` to True and everything else to False 
    - the samples will be saved in `model_sample_path`
	

Plotting the samples from the model:

 - to plot the psychometric curve, set `run_plot_psychometric_individual_samples` to True 
    - `run_plot_prop_choice_dist` and `run_plot_psychom_metric_comparison_subjective_change` are out of date and should not be used 
 - to plot the chronometric curve, set `run_plot_chronometric` to True 
 - to plot the comparison between model and mouse reaction times, set `run_plot_compare_model_and_mouse_dist` to True 
 
 
 Comparison of fitting performance across models and mice (work in progress): 
 
  - go to `normative_hmm_main.py`
  
  
 
	
	

 


 

