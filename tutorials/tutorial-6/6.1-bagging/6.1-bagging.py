# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:08:04 2020

@author: rubenandrebarreiro
"""

# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import SciKit-Learn as skl
import sklearn as skl

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import System Python's Library
import sys

# Append the Path "../" to the System's Path
sys.path.append('../')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The file of the Dataset for Training
dataset_train_file = "../files/data/T6-train.txt"

# The file of the Dataset for Validation
dataset_valid_file = "../files/data/T6-valid.txt"


# Load the Data for Training Dataset with NumPy function loadtxt
dataset_train_not_random = np.loadtxt(dataset_train_file)

# Load the Data for Validation Dataset with NumPy function loadtxt
dataset_valid_not_random = np.loadtxt(dataset_valid_file)


# Shuffle the Training Dataset, not randomized
dataset_train_random = skl.utils.shuffle(dataset_train_not_random)

# Shuffle the Validation Dataset, not randomized
dataset_valid_random = skl.utils.shuffle(dataset_valid_not_random)


# Number of Replicas
num_replicas = 200

# The Bootstrap Ensemble Method
def bootstrap(samples, data):
    
    train_sets = np.zeros((samples, data.shape[0], data.shape[1]))
    
    for sample in range(samples):
        ix = np.random.randint(data.shape[0], size=data.shape[0])
        train_sets[sample,:] = data[ix,:]
        
    return train_sets



# The Training Sets, with Replicas randomly generated,
# from the Ensemble Bagging Method
train_sets = bootstrap(num_replicas, dataset_train_random)


single_preds_train = np.zeros((len(dataset_train_random), len(dataset_train_random)))
single_preds_valid = np.zeros((len(dataset_valid_random), len(dataset_valid_random)))

bagging_preds_train = np.zeros((num_replicas, len(dataset_train_random)))
bagging_preds_valid = np.zeros((num_replicas, len(dataset_valid_random)))


single_train_errors_avgs = np.zeros((15, 2))
single_valid_errors_avgs = np.zeros((15, 2))

bagging_train_errors_avgs = np.zeros((15, 2))
bagging_valid_errors_avgs = np.zeros((15, 2))


for degree in range(15):
    
    
    for ix_single in range(len(dataset_train_random)):
        
        single_coefs = np.polyfit(dataset_train_random[:, 0], dataset_train_random[:, 1], degree)
        
        single_preds_train[ix_single, :] = np.polyval(single_coefs, dataset_train_random[:, 0])
        single_preds_valid[ix_single, :] = np.polyval(single_coefs, dataset_valid_random[:, 0])
    
    
    for ix_bagging in range(num_replicas):
        
        bagging_coefs = np.polyfit(train_sets[ix_bagging, :, 0], train_sets[ix_bagging, :, 1], degree)
        
        bagging_preds_train[ix_bagging,:] = np.polyval(bagging_coefs, dataset_train_random[:, 0])
        bagging_preds_valid[ix_bagging,:] = np.polyval(bagging_coefs, dataset_valid_random[:, 0])
    
    
    single_mean_preds_train = np.mean(single_preds_train, axis=0)
    single_mean_preds_valid = np.mean(single_preds_valid, axis=0)
    
    single_train_errors = (single_mean_preds_train - dataset_train_random[:, 1])**2
    single_valid_errors = (single_mean_preds_valid - dataset_valid_random[:, 1])**2
    
    single_train_errors_avgs[degree, 0] = degree
    single_train_errors_avgs[degree, 1] = ( np.sum(single_train_errors) / len(single_train_errors) )
    
    single_valid_errors_avgs[degree, 0] = degree
    single_valid_errors_avgs[degree, 1] = ( np.sum(single_valid_errors) / len(single_valid_errors) )    
    
    
    bagging_mean_preds_train = np.mean(bagging_preds_train, axis=0)
    bagging_mean_preds_valid = np.mean(bagging_preds_valid, axis=0)
    
    bagging_train_errors = (bagging_mean_preds_train - dataset_train_random[:, 1])**2
    bagging_valid_errors = (bagging_mean_preds_valid - dataset_valid_random[:, 1])**2
    
    bagging_train_errors_avgs[degree, 0] = degree
    bagging_train_errors_avgs[degree, 1] = ( np.sum(bagging_train_errors) / len(bagging_train_errors) )
    
    bagging_valid_errors_avgs[degree, 0] = degree
    bagging_valid_errors_avgs[degree, 1] = ( np.sum(bagging_valid_errors) / len(bagging_valid_errors) )    
    
    
# The Function to plot the Single/Ensemble Bagging
# Training and Validation Errors 
def plot_single_and_bagging_training_validation_errors(single_train_errors_avgs, single_valid_errors_avgs, bagging_train_errors_avgs, bagging_valid_errors_avgs):
    
    # Initialise the Plot
    plt.figure(figsize=(14, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Single/Ensemble Bagging Training and Validation Errors
    plt.plot(single_train_errors_avgs[:,0], single_train_errors_avgs[:,1], '-', color="blue", label="Single Training Error")
    plt.plot(single_valid_errors_avgs[:,0], single_valid_errors_avgs[:,1], '-', color="orange", label="Single Validation Error")
    plt.plot(bagging_train_errors_avgs[:,0], bagging_train_errors_avgs[:,1], '-', color="green", label="Ensemble/Bagging Training Error")
    plt.plot(bagging_valid_errors_avgs[:,0], bagging_valid_errors_avgs[:,1], '-', color="red", label="Ensemble/Bagging Validation Error")
    
    plt.legend(loc="lower left", frameon=True)
    
    # Set the axis for the Plot
    plt.axis([-1, 15, 0.0, 1.0])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("Polynomial Degree")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Single/Ensemble Bagging Training and Validation Errors")
    
    # Set the Title of the Plot
    plt.title('Single/Ensemble Bagging Training and Validation Errors')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/single-ensemble-bagging-training-and-validation-errors.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
plot_single_and_bagging_training_validation_errors(single_train_errors_avgs, single_valid_errors_avgs, bagging_train_errors_avgs, bagging_valid_errors_avgs)