# ML_Project

## Authors 
* Diana Petrescu 
* Patrik Wagner
* Thevie Mortiniera

This is the repository of our Mini Project in the Machine Learning Course.
Our job was to predict if a signature results from a Higgs Boson (signal) or some other process (background). We were given two data sets :
- a training set consisting 250000 rows (events) and 30 attributes (signature of the event)
- a test set with the features and missing values for the variable we want to predict.
The current file explains the organisation of the project folder, the description of the functions in the different python scripts and other related files.

## Report
it contains detailed explanations of our scientific approach with the results.

## cross_validation
This file contains multiples methods allowing us to run cross validation on our predictions.
- build_k_indices
- cross_validation
- cross_validation_demo
- cross_validation_ridge
- cross_validation_ridge_demo
- predict_test


## helpers

## implementations
This file contains several regression functions we used in our predictions and some helper functons.
- calculate_mse
- calculate_rmse
- compute_loss
- batch_iter
- compute_gradient
- least_squares_GD
- least_squares_SGD
- least_squares
- ridge_regression
- sigmoid
- calculate_loss_logistic
- calculate_gradient_logistic
- learning_by_gradient_descent_logistic
- logistic_regression
- penalized_logistic_regression
- learning_by_penalized_gradient_logistic
- reg_logistic_regression

## model_selection

## preprocessing
This file contains methods for preprocessing the data, such as data cleaning and feature engineering techniques. It contains :
- standardize 
- de_standardize
- feature_selection
- column_weighting
- inv_log_f
- process_data
- na : determine if a vector contains undefined values
- process_data2
- process_data3

## proj1_helpers
This file is used to load the data set and create the csv submission file for Kaggle. It contains 3 functions :
- load_csv_data
- predict_labels
- create_csv_submission

## visualization
This file contains different visualisation methods we used during our tests to see how well our models performed.
- cross_validation_visualization
- cross_validation_visualization_ridge
- bias_variance_decomposition_visualization
- visualization_classification

## run
In order to run this script you can run the following command in the command line, provided that you have Train.csv and Test.csv files in Data folder.

Command : python3 run.py Data/Train.csv Data/Test.csv

It will produce, as output, a csv file containing our predictions as the ones we submitted on Kaggle






