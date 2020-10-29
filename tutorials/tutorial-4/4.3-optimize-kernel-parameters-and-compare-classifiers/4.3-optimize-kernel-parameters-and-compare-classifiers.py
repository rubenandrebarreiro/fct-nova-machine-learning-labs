# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:49:07 2020

@author: rubenandrebarreiro
"""

# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import Math Python's Library as mathematics
import math as mathematics

# Import SciKit-Learn as skl
import sklearn as skl

# Import Train/Test Split,
# from the SciKit-Learn's Model Selection Module,
# as split_train_test_sets
from sklearn.model_selection import train_test_split as split_train_test_sets

# Import Model Selection Sub-Module, from SciKit-Learn Python's Library,
# as skl_model_selection 
from sklearn import model_selection as skl_model_selection

# Import Support Vector Classifier,
# from the SciKit-Learn's Support Vector Machine Module,
# as support_vector_machine
from sklearn.svm import SVC as support_vector_machine

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import System Python's Library
import sys

# Append the Path "../" to the System's Path
sys.path.append('../')

# Import the Support Vector Machine Plot Functions,
# from the Customised T4_Aux Python's Library
from files.T4aux import plot_svm_mark_wrong_x as plot_support_vector_machine_corrects_o_wrongs_x

# Import Warnings
import warnings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Ignore Warnings
warnings.filterwarnings("ignore")

# The file of the Dataset
dataset_file = "../files/data/T4data.txt"

# Load the Data for Dataset with NumPy function loadtxt
dataset_not_random = np.loadtxt(dataset_file, delimiter="\t")    


# Shuffle the Dataset, not randomized
dataset_random = skl.utils.shuffle(dataset_not_random)


# Select the Classes of the Dataset, randomized
ys_dataset_classes = dataset_random[:,-1]                                 

# Select the Features of the Dataset, randomized
xs_dataset_features = dataset_random[:,0:-1]                              

# The size of the Data for Dataset, randomized
dataset_size = len(xs_dataset_features)


# Computing the Means of the Dataset, randomized
dataset_means = np.mean(xs_dataset_features, axis=0)

# Computing the Standard Deviations of the Dataset, randomized
dataset_stdevs = np.std(xs_dataset_features, axis=0)                                


# Standardize the Dataset, randomized
xs_dataset_features_std = ( ( xs_dataset_features - dataset_means ) / dataset_stdevs )

# Split the Dataset Standardized, into Training and Testing Sets,
# by a ratio of 50% for each one
xs_train_features_std, xs_test_features_std, ys_train_classes, ys_test_classes = split_train_test_sets(xs_dataset_features_std, ys_dataset_classes, test_size=0.5)

# The number of Samples of the Testing Set
num_samples_test_set = len(xs_test_features_std) 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Constants #1

# The Number of Features (i.e., 2 Features)
NUM_FEATURES = xs_train_features_std.shape[1]

# The Number of Folds, for Stratified K Folds, in Cross-Validation
NUM_FOLDS = 10


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The Function to Plot the Training and Validation, for the Support Vector Machine
def plot_train_valid_error_support_vector_machine(support_vector_machine_kernel, train_error_values, valid_error_values):
    
    # Initialise the Plot
    plt.figure(figsize=(8, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Training and Validation Errors
    plt.plot(train_error_values[:,0], train_error_values[:,1],'-', color="dodgerblue")
    plt.plot(valid_error_values[:,0], valid_error_values[:,1],'-', color="orange")
    
    # Set the axis for the Plot
    plt.axis([min(valid_error_values[:,0]), max(valid_error_values[:,0]), min(train_error_values[:,1]), max(valid_error_values[:,1])])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("log(C)")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Training/Validation Errors")
    
    
    support_vector_machine_kernel_name = ""
    
    if(support_vector_machine_kernel[0] == "poly"):
        support_vector_machine_kernel_name = "Polynomial"
        
    if(support_vector_machine_kernel[0] == "sigmoid"):
        support_vector_machine_kernel_name = "Sigmoid"
        
    if(support_vector_machine_kernel[0] == "rbf"):
        support_vector_machine_kernel_name = "Gaussian RBF"
    
    # Set the Title of the Plot
    plt.title('Support Vector Machine, with Kernel=\"{}\", varying the C parameter\n\nTraining Error (Blue) / Cross-Validation Error (Orange)'.format(support_vector_machine_kernel_name))
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/{}-training-validation-errors.png'.format(support_vector_machine_kernel[0]), dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()

# The Function to Compute and Return the Errors for Training and Validation Sets,
# for the Support Vector Machines Classifier
def compute_svm_kernel_errors(xs, ys, train_idx, valid_idx, current_support_vector_machine_kernel, c_param_value, num_features):
    
    support_vector_machine_classifier = support_vector_machine(kernel=current_support_vector_machine_kernel[0], degree=current_support_vector_machine_kernel[1], gamma=current_support_vector_machine_kernel[2], coef0=current_support_vector_machine_kernel[3], C=c_param_value)
        
    # Fit the Support Vector Machine Classifier with the Data from the Training Set
    support_vector_machine_classifier.fit(xs[train_idx,:2], ys[train_idx])
    
    # Set the Filename for the Plot of the Data, for the current Kernel and its configurations
    plot_support_vector_machine_image_filename = "imgs/support-vector-machine-plot-c-{}-{}-kernel.png".format(c_param_current_value, current_support_vector_machine_kernel[0])
    
    
    # Compute the Training Set's Accuracy (Score), for the current Support Vector Machine
    support_vector_machine_accuracy_train = support_vector_machine_classifier.score(xs[train_idx], ys[train_idx])
    
    # Compute the Validation Set's Accuracy (Score), for the current Support Vector Machine
    support_vector_machine_accuracy_valid = support_vector_machine_classifier.score(xs[valid_idx], ys[valid_idx])
    
    
    # Compute the Training Error, regarding its Accuracy (Score)
    support_vector_machine_train_error = ( 1 - support_vector_machine_accuracy_train )
    
    # Compute the Validation Error, regarding its Accuracy (Score)
    support_vector_machine_valid_error = ( 1 - support_vector_machine_accuracy_valid )                  
    
    
    # Prepare Data from the Training Set, for the Plot Function
        
    # Create a Matrix, with num_samples_train rows,
    # and 3 columns (2 features + 1 class)
    dataset_train = np.zeros((len(xs[train_idx,:2]), 3))
        
    # Fill the Data from the Training Set, for the Plot Function
    dataset_train[:,0:2] = xs[train_idx,:2][:,:]
    dataset_train[:,2] = ys[train_idx]
        
    plot_support_vector_machine_corrects_o_wrongs_x(False, dataset_train, support_vector_machine_classifier, current_support_vector_machine_kernel[0], plot_support_vector_machine_image_filename, c_param_current_value)

        
    # Return the Training and Validation Errors, for the Logistic Regression
    return support_vector_machine_train_error, support_vector_machine_valid_error


# The Function to Compute and Return the Errors for Training and Validation Sets,
# for the Support Vector Machines Classifier
def plot_svm_kernel_best_c(current_support_vector_machine_kernel, best_c_param_value, num_features):
    
    support_vector_machine_classifier = support_vector_machine(kernel=current_support_vector_machine_kernel[0], degree=current_support_vector_machine_kernel[1], gamma=current_support_vector_machine_kernel[2], coef0=current_support_vector_machine_kernel[3], C=best_c_param_value)
        
    # Fit the Support Vector Machine Classifier with the Data from the Training Set
    support_vector_machine_classifier.fit(xs_train_features_std, ys_train_classes)
    
    # Set the Filename for the Plot of the Data, for the current Kernel and its configurations
    plot_support_vector_machine_image_filename = "imgs/support-vector-machine-plot-best-c-{}-{}-kernel.png".format(best_c_param_value, current_support_vector_machine_kernel[0])
    
    
    # Prepare Data from the Training Set, for the Plot Function
        
    # Create a Matrix, with num_samples_train rows,
    # and 3 columns (2 features + 1 class)
    dataset_train = np.zeros((len(xs_train_features_std), 3))
        
    # Fill the Data from the Training Set, for the Plot Function
    dataset_train[:,0:2] = xs_train_features_std[:,:]
    dataset_train[:,2] = ys_train_classes
        
    plot_support_vector_machine_corrects_o_wrongs_x(True, dataset_train, support_vector_machine_classifier, current_support_vector_machine_kernel[0], plot_support_vector_machine_image_filename, best_c_param_value)


# The Function to Estimate the True/Test Error of the Testing Set,
# for the Support Vector Machine Classifier
def estimate_svm_kernel_true_test_error(xs_train, ys_train, xs_test, ys_test, num_features, current_support_vector_machine_kernel, best_c_param_value=1e12):
    
    support_vector_machine_classifier = support_vector_machine(kernel=current_support_vector_machine_kernel[0], degree=current_support_vector_machine_kernel[1], gamma=current_support_vector_machine_kernel[2], coef0=current_support_vector_machine_kernel[3], C=best_c_param_value)
        
    # Fit the Support Vector Machine Classifier with the Data from the whole Training Set (together with Validation Set)
    support_vector_machine_classifier.fit(xs_train, ys_train)

        
    # Estimate the Testing Set's Accuracy (Score), for the current Support Vector Machine
    support_vector_machine_estimated_accuracy_test = support_vector_machine_classifier.score(xs_test, ys_test)
        
    # Estimate the Testing Error, regarding its Accuracy (Score)
    support_vector_machine_estimated_true_test_error = ( 1 - support_vector_machine_estimated_accuracy_test )
    
    
    # Predict the Probabilities of the Features of the Testing Set, belongs to a certain Class
    ys_classes_support_vector_machine_prediction_xs_test = support_vector_machine_classifier.predict(xs_test)
    
    # The Number of Samples, from the Testing Set 
    num_samples_test_set = len(xs_test)                                        
    
    # The Real Number of Incorrect Predictions, regarding the Support Vector Machine Classifier
    support_vector_machine_num_incorrect_predictions = 0                                       
    
    # For each Sample, from the Testing Set
    for current_sample_test in range(num_samples_test_set):
        
        # If the Prediction/Classification of the Class for the current Sample, of the Testing Set is different from the Real Class of the same,
        # it's considered an Real Error in Prediction/Classification, regarding the Logistic Regression Classifier
        if(ys_classes_support_vector_machine_prediction_xs_test[current_sample_test] != ys_test[current_sample_test] ):
            support_vector_machine_num_incorrect_predictions += 1
            
    
    # Return the Predictions of the Samples,
    # the Real Number of Incorrect Predictions and the Estimated True/Test Error, for the Logistic Regression Classifier
    return ys_classes_support_vector_machine_prediction_xs_test, support_vector_machine_num_incorrect_predictions, support_vector_machine_estimated_true_test_error



def aproximate_normal_test(num_real_errors, prob_making_error, num_samples_test_set):

    prob_errors_in_test_set = ( num_real_errors / num_samples_test_set )
    prob_not_errors_in_test_set = ( 1 - prob_errors_in_test_set )

    NormalTest_deviation = mathematics.sqrt( num_samples_test_set * prob_errors_in_test_set * prob_not_errors_in_test_set )
    
    NormalTest_LowerDeviation = ( -1 * 1.96 * NormalTest_deviation )
    NormalTest_UpperDeviation = ( 1.96 * NormalTest_deviation )

    return NormalTest_LowerDeviation, NormalTest_UpperDeviation



def mc_nemar_test(predict_classes_xs_test_1, predict_classes_xs_test_2):
    
    num_samples_test_set = len(xs_test_features_std)
    
    first_wrong_second_right = 0
    first_right_second_wrong = 0
    
    for current_sample_test in range(num_samples_test_set):
        
        if( ( predict_classes_xs_test_1[current_sample_test] != ys_test_classes[current_sample_test] ) and ( predict_classes_xs_test_2[current_sample_test] == ys_test_classes[current_sample_test] ) ):
            first_wrong_second_right += 1
        
        if( ( predict_classes_xs_test_1[current_sample_test] == ys_test_classes[current_sample_test] ) and ( predict_classes_xs_test_2[current_sample_test] != ys_test_classes[current_sample_test] ) ):
            first_right_second_wrong += 1
    
    
    mc_nemar_test_dividend = ( ( abs(first_wrong_second_right - first_right_second_wrong) - 1) ** 2 )
    mc_nemar_test_divider = ( first_wrong_second_right + first_right_second_wrong )
    
    mc_nemar_test_value = ( mc_nemar_test_dividend / mc_nemar_test_divider )
    
    
    return mc_nemar_test_value


# Setting the Kernels and their Parameters,
# for the Support Vector Machines Classifiers

# 1. Polynomial,
#    K(x; y) = (xT y + r)d,
#    with degree 3, gamma of 0.5 and r of 0
#    (the r value is set in the coef0 parameter and is 0 by default).
# 2. Sigmoid,
#    K(x; y) = tanh(xT y + r),
#    with gamma of 0.5 and r of -2.
# 3. Gaussian RBF,
#    K(x; y) = exp(􀀀jjx 􀀀 yjj2), with gamma of 0.5.

# Parameters of the Kernels of the Support Vector Machine Classifiers
# - 1) Name of the Kernel;
# - 2) Degree of the Kernel;
# - 3) Gamma of the Kernel;
# - 4) r value (coef_0) of the Kernel;
support_vector_machine_kernels = [ ["poly", 3.0, 0.5, 0.0],
                                   ["sigmoid", 0.0, 0.5, -2.0],
                                   ["rbf", 0.0, 0.5, 0.0] ]

support_vector_machine_kernels_best_c = np.zeros(3)
support_vector_machine_kernels_best_valid_error = np.zeros(3)

support_vector_machine_kernels_predictions_xs_test = []
support_vector_machine_kernels_num_incorrect_predictions = np.zeros(3)
support_vector_machine_kernels_estimated_true_test_error = np.zeros(3)

for current_support_vector_machine_kernel_index in range(len(support_vector_machine_kernels)):

    
    svm_kernel_name = ""

    if(support_vector_machine_kernels[current_support_vector_machine_kernel_index][0] == "poly"):
        svm_kernel_name = "POLYNOMIAL"

    if(support_vector_machine_kernels[current_support_vector_machine_kernel_index][0] == "sigmoid"):
        svm_kernel_name = "SIGMOID"

    if(support_vector_machine_kernels[current_support_vector_machine_kernel_index][0] == "rbf"):
        svm_kernel_name = "GAUSSIAN RBF"

    
    print("#############################################################")
    print("     {}) {}".format((current_support_vector_machine_kernel_index + 1), svm_kernel_name))
    print("#############################################################")
    print("\n")    

    # The K Folds Combinations Model, for the Stratified K Folds process
    k_folds = skl_model_selection.StratifiedKFold(n_splits = NUM_FOLDS)     

    # The Best Regularization Parameter C found, for Support Vector Machine    
    svm_kernel_best_c = 1e10                                    
    
    # The Best Average of the Validation Error, for Support Vector Machine
    svm_kernel_best_valid_error_avg_folds = 1e10                
    
    
    # The Initial/Current Regularization Parameter C (i.e., 1e-1)
    c_param_current_value = 1e-1
    
    # The Final Regularization Parameter C (i.e., 1e4)    
    c_param_final_value = 1e4
    
    
    # The Number of Steps, for varying the C Regularization Paramater
    num_steps_c_regularization = 0
    
    # While the current C Regularization Paramater be valid
    while(c_param_current_value <= c_param_final_value):
        
        # Double the current value of C Regularization Paramater
        c_param_current_value = 2 * c_param_current_value
        
        # Increment the Number of Steps, for varying the C Regularization Paramater
        num_steps_c_regularization = num_steps_c_regularization + 1
        
        
    # Reset the Initial/Current Regularization Parameter C (i.e., 1e-1)
    c_param_current_value = 1e-1
    
    
    # The Values of Training and Validation Errors, for Support Vector Machine 
    svm_kernel_train_error_values = np.zeros((num_steps_c_regularization,2))        
    svm_kernel_valid_error_values = np.zeros((num_steps_c_regularization,2))
    
    
    # Reset the Number of Steps, for varying the C Regularization Paramater
    num_steps_c_regularization = 0
    
    # Select the current Support Vector Machine and its configuration of parameters
    current_support_vector_machine_kernel = support_vector_machine_kernels[current_support_vector_machine_kernel_index]
    
    
    # While the current C Regularization Paramater be valid
    while(c_param_current_value <= c_param_final_value):

        print("-------------------------------------------------------------")
        print("\n")
        print("Trying the \"{}\" Kernel, to the Support Vector Machine Classifier, with the following Parameters:".format(current_support_vector_machine_kernel[0]))
        print("- C Regularization Parameter: {} ;".format(c_param_current_value))
        print("- Degree: {} (only considered in \"Poly\" Kernel) ;".format(current_support_vector_machine_kernel[1]))
        print("- Gamma: {} ;".format(current_support_vector_machine_kernel[2]))
        print("- r value (coef_0 parameter): {} (only considered in \"Poly\" and \"Sigmoid\" Kernels) ;".format(current_support_vector_machine_kernel[3]))
        print("\n")
        
        # The sum of the Training and Validation Errors, for Support Vector Machines
        svm_kernel_train_error_sum = 0                                                      
        svm_kernel_valid_error_sum = 0
        
        # The average of the Training and Validation Errors, by the Number of Folds,
        # for Support Vector Machines
        svm_kernel_train_error_avg_folds = 0
        svm_kernel_valid_error_avg_folds = 0
        
        # The loop for all the combinations of K Folds, in the Stratified K Folds process
        for train_idx, valid_idx in k_folds.split(ys_train_classes, ys_train_classes):
            
            # Compute the Training and Validation Errors, for Support Vector Machines
            svm_kernel_train_error, svm_kernel_valid_error = compute_svm_kernel_errors(xs_train_features_std, ys_train_classes, train_idx, valid_idx, current_support_vector_machine_kernel, c_param_current_value, NUM_FEATURES)
            
            # Sum the current Training and Validation Errors to the Sums of them
            svm_kernel_train_error_sum += svm_kernel_train_error
            svm_kernel_valid_error_sum += svm_kernel_valid_error
            
            
        # Compute the Average of the Sums of the Training and Validation Errors, by the Total Number of Folds 
        svm_kernel_train_error_avg_folds = (svm_kernel_train_error_sum / NUM_FOLDS)
        svm_kernel_valid_error_avg_folds = (svm_kernel_valid_error_sum / NUM_FOLDS)

        print("Computed Errors from Cross-Validation:")
        print("- Training Error = {} ;".format(svm_kernel_train_error_avg_folds))
        print("- Validation Error = {} ;".format(svm_kernel_valid_error_avg_folds))
        print("\n")


        # Updates the Best Validation Error and also, the Best Regularization C Parameter
        if(svm_kernel_best_valid_error_avg_folds > svm_kernel_valid_error_avg_folds):
            
            svm_kernel_best_valid_error_avg_folds = svm_kernel_valid_error_avg_folds
            svm_kernel_best_c = c_param_current_value
            
            support_vector_machine_kernels_best_valid_error[current_support_vector_machine_kernel_index] = svm_kernel_valid_error_avg_folds
            support_vector_machine_kernels_best_c[current_support_vector_machine_kernel_index] = c_param_current_value
        
        
        # Store the Values for x and y, for all the Training Error values
        # for the Plot of the Training Errors, as a Function of Logarithm of the C Parameter
        svm_kernel_train_error_values[num_steps_c_regularization, 0] = np.log(c_param_current_value)        
        svm_kernel_train_error_values[num_steps_c_regularization, 1] = svm_kernel_train_error_avg_folds         
        
        # Store the Values for x and y, for all the Validation Error values
        # for the Plot of the Validation Errors, as a Function of Logarithm of the C Parameter
        svm_kernel_valid_error_values[num_steps_c_regularization, 0] = np.log(c_param_current_value)
        svm_kernel_valid_error_values[num_steps_c_regularization, 1] = svm_kernel_valid_error_avg_folds
        
        # Double the current value of C Regularization Paramater
        c_param_current_value = 2 * c_param_current_value

        # Increment the Number of Steps, for varying the C Regularization Paramater
        num_steps_c_regularization = num_steps_c_regularization + 1
    
        print("-------------------------------------------------------------")
        print("\n")

    # Plot the Training Set's Prediction, for the Support Vector Machine Classifier,
    # with the best C Regularization Parameter found
    plot_svm_kernel_best_c(current_support_vector_machine_kernel, support_vector_machine_kernels_best_c[current_support_vector_machine_kernel_index], NUM_FEATURES)
    
    # Plot the Training and Validation Errors, for the Support Vector Machine Classifier,
    # varying the C Regularization Parameter
    plot_train_valid_error_support_vector_machine(current_support_vector_machine_kernel, svm_kernel_train_error_values, svm_kernel_valid_error_values)
    
    
    svm_prediction_xs_test, svm_num_incorrect_predictions, svm_estimated_true_test_error = estimate_svm_kernel_true_test_error(xs_train_features_std, ys_train_classes, xs_test_features_std, ys_test_classes, NUM_FEATURES, current_support_vector_machine_kernel, support_vector_machine_kernels_best_c[current_support_vector_machine_kernel_index])
    
    support_vector_machine_kernels_predictions_xs_test.append(svm_prediction_xs_test)
    support_vector_machine_kernels_num_incorrect_predictions[current_support_vector_machine_kernel_index] = svm_num_incorrect_predictions
    support_vector_machine_kernels_estimated_true_test_error[current_support_vector_machine_kernel_index] = svm_estimated_true_test_error
        
    


# Computes the Aproximate Normal Test, for the Support Vector Machine Classifier,
# with 'Polynomial' Kernel
svm_poly_normalTest_LowDeviation, svm_poly_normalTest_UpperDeviation = aproximate_normal_test(support_vector_machine_kernels_num_incorrect_predictions[0], support_vector_machine_kernels_estimated_true_test_error[0], num_samples_test_set)

# Computes the Aproximate Normal Test, for the Support Vector Machine Classifier,
# with 'Sigmoid' Kernel
svm_sigmoid_normalTest_LowDeviation, svm_sigmoid_normalTest_UpperDeviation = aproximate_normal_test(support_vector_machine_kernels_num_incorrect_predictions[1], support_vector_machine_kernels_estimated_true_test_error[1], num_samples_test_set)

# Computes the Aproximate Normal Test, for the Support Vector Machine Classifier,
# with 'Gaussian RBF' Kernel
svm_rbf_normalTest_LowDeviation, svm_rbf_normalTest_UpperDeviation = aproximate_normal_test(support_vector_machine_kernels_num_incorrect_predictions[2], support_vector_machine_kernels_estimated_true_test_error[2], num_samples_test_set)


print("Best Validation Errors found, for each Support Vector Machine's Kernel:")
print("- Polynomial = {}".format(support_vector_machine_kernels_best_valid_error[0]))
print("- Sigmoid = {}".format(support_vector_machine_kernels_best_valid_error[1]))
print("- Gaussian RBF = {}".format(support_vector_machine_kernels_best_valid_error[2]))
print("\n\n")


print("Best Regularization C Parameters found, for each Support Vector Machine's Kernel:")
print("- Polynomial = {}".format(support_vector_machine_kernels_best_c[0]))
print("- Sigmoid = {}".format(support_vector_machine_kernels_best_c[1]))
print("- Gaussian RBF = {}".format(support_vector_machine_kernels_best_c[2]))
print("\n\n")


print("Number of Incorrect Predictions, for each Support Vector Machine's Kernel:")
print("- Polynomial = {}".format(support_vector_machine_kernels_num_incorrect_predictions[0]))
print("- Sigmoid = {}".format(support_vector_machine_kernels_num_incorrect_predictions[1]))
print("- Gaussian RBF = {}".format(support_vector_machine_kernels_num_incorrect_predictions[2]))
print("\n\n")


print("Estimated True/Test Errors, for each Support Vector Machine's Kernel:")
print("- Polynomial = {}".format(support_vector_machine_kernels_estimated_true_test_error[0]))
print("- Sigmoid = {}".format(support_vector_machine_kernels_estimated_true_test_error[1]))
print("- Gaussian RBF = {}".format(support_vector_machine_kernels_estimated_true_test_error[2]))
print("\n\n")


print("Approximate Normal Test Intervals, with Confidence Level of 95%, for each Support Vector Machine's Kernel:")
print("- Polynomial = [ {} ; {} ]".format( ( support_vector_machine_kernels_num_incorrect_predictions[0] + svm_poly_normalTest_LowDeviation ), ( support_vector_machine_kernels_num_incorrect_predictions[0] + svm_poly_normalTest_UpperDeviation ) ))
print("- Sigmoid = [ {} ; {} ]".format( ( support_vector_machine_kernels_num_incorrect_predictions[1] + svm_sigmoid_normalTest_LowDeviation ), ( support_vector_machine_kernels_num_incorrect_predictions[1] + svm_sigmoid_normalTest_UpperDeviation ) ))
print("- Gaussian RBF = [ {} ; {} ]".format( ( support_vector_machine_kernels_num_incorrect_predictions[2] + svm_rbf_normalTest_LowDeviation ), ( support_vector_machine_kernels_num_incorrect_predictions[2] + svm_rbf_normalTest_UpperDeviation ) ))
print("\n\n")


mc_nemar_test_svm_poly_vs_svm_sigmoid_value = mc_nemar_test(support_vector_machine_kernels_predictions_xs_test[0], support_vector_machine_kernels_predictions_xs_test[1])
mc_nemar_test_svm_poly_vs_svm_rbf_value = mc_nemar_test(support_vector_machine_kernels_predictions_xs_test[0], support_vector_machine_kernels_predictions_xs_test[2])
mc_nemar_test_svm_sigmoid_vs_svm_rbf_value = mc_nemar_test(support_vector_machine_kernels_predictions_xs_test[1], support_vector_machine_kernels_predictions_xs_test[2])


#---------------------------------------- SVM: Polynomial vs SVM: Sigmoid ------------------------------------------------------#

print("Result of the McNemar Test #1: SVM: Polynomial vs. SVM: Sigmoid:")
print("- {}".format(mc_nemar_test_svm_poly_vs_svm_sigmoid_value))
    
if(mc_nemar_test_svm_poly_vs_svm_sigmoid_value >= 3.84):        
    print("\n")
    print("The SVM: Polynomial and SVM: Sigmoid, ARE significantly different!!!")    
else:    
    print("\n")
    print("The SVM: Polynomial and SVM: Sigmoid, ARE NOT significantly different!!!")

print("\n\n")
    
#---------------------------------------- SVM: Polynomial vs SVM: Sigmoid ------------------------------------------------------#
    
#---------------------------------------- SVM: Polynomial vs SVM: Gaussian RBF -------------------------------------------------#

print("Result of the McNemar Test #2: SVM: Polynomial vs. SVM: Gaussian RBF:")
print("- {}".format(mc_nemar_test_svm_poly_vs_svm_rbf_value))
    
if(mc_nemar_test_svm_poly_vs_svm_rbf_value >= 3.84):        
    print("\n")
    print("The SVM: Polynomial and SVM: Gaussian RBF, ARE significantly different!!!")    
else:    
    print("\n")
    print("The SVM: Polynomial and SVM: Gaussian RBF, ARE NOT significantly different!!!")

print("\n\n")

#---------------------------------------- SVM: Polynomial vs SVM: Gaussian RBF -------------------------------------------------#
    
#---------------------------------------- SVM: Sigmoid vs SVM: Gaussian RBF ------------------------------------------------------#

print("Result of the McNemar Test #3: SVM: Sigmoid vs. SVM: Gaussian RBF:")
print("- {}".format(mc_nemar_test_svm_sigmoid_vs_svm_rbf_value))
    
if(mc_nemar_test_svm_sigmoid_vs_svm_rbf_value >= 3.84):        
    print("\n")
    print("The SVM: Sigmoid and SVM: Gaussian RBF, ARE significantly different!!!")    
else:    
    print("\n")
    print("The SVM: Sigmoid and SVM: Gaussian RBF, ARE NOT significantly different!!!")

print("\n\n")

#---------------------------------------- SVM: Sigmoid vs SVM: Gaussian RBF ------------------------------------------------------#
    