# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:02:04 2020

@author: rubenandrebarreiro
"""

# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import SciKit-Learn as skl
import sklearn as skl

# Import Train/Test Split,
# from the SciKit-Learn's Model Selection Module,
# as split_train_test_sets
from sklearn.model_selection import train_test_split as split_train_test_sets

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import Warnings
import warnings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Ignore Warnings
warnings.filterwarnings("ignore")

# The file of the Dataset
dataset_file = "../files/data/T5data.txt"

# Load the Data for Dataset with NumPy function loadtxt
dataset_not_random = np.loadtxt(dataset_file)

# Shuffle the Dataset, not randomized
dataset_random = skl.utils.shuffle(dataset_not_random)

# Split the Dataset, into Training and Testing Sets,
# by a ratio of 50% for each one
train_set_data, test_set_data = split_train_test_sets(dataset_random, test_size=0.5)    


# Function to Compute the Bias and Variance of the Testing Set
def compute_bias_and_variance_test_set(poly_degree, replicas, test_set_data):

    for replica_index in range(replicas.shape[0]):
        coefs = np.polyfit(replicas[replica_index,:,0], replicas[replica_index,:,1], poly_degree)
        predictions[replica_index,:] = np.polyval(coefs, test_set_data[:,0])
        
    mean_predictions = np.mean(predictions, axis = 0)
    bias_test_set = np.mean((test_set_data[:,1] - mean_predictions)**2)
    
    predictions_squared = predictions**2
    mean_square = np.mean(predictions_squared, axis = 0)
    variance_test_set = np.mean(mean_square - mean_predictions**2)
    
    return bias_test_set, variance_test_set


# The Function to Biases and Variances of the Testing Set
def plot_bias_and_variance_test_set(biases_test_set, variances_test_set):
    
    # Initialise the Plot
    plt.figure(figsize=(8, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Biases and Variances of the Testing Set
    plt.plot(biases_test_set[:,0], biases_test_set[:,1], '-', color="red", label="Bias")
    plt.plot(variances_test_set[:,0], variances_test_set[:,1], '-', color="blue", label="Variance")
    
    plt.legend(loc="upper right", frameon=True)
    
    # Set the axis for the Plot
    #plt.axis([min(biases_test_set[:,0]), max(biases_test_set[:,0]), min(biases_test_set[:,1]), max(variances_test_set[:,1])])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("Polynomial Degree")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Bias/Variance Values")
    
    # Set the Title of the Plot
    plt.title('Bias (Red) / Variance (Blue) of the Testing Set')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/bias-and-variance-testing-set.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


replicas = np.zeros((500,100,2))

for sample_x_train in range(500):
    x_index = np.random.randint(100, size = 100)
    replicas[sample_x_train, :] = train_set_data[x_index, :]


predictions = np.zeros((500,100))

poly_degree_biases_test_set = np.zeros((14,2))
poly_degree_variances_test_set = np.zeros((14,2))

for poly_degree in range(1,15):

    bias_test_set, variance_test_set = compute_bias_and_variance_test_set(poly_degree+1, replicas, test_set_data)
    
    poly_degree_biases_test_set[poly_degree-1, 0] = poly_degree
    poly_degree_biases_test_set[poly_degree-1, 1] = bias_test_set
    
    poly_degree_variances_test_set[poly_degree-1, 0] = poly_degree
    poly_degree_variances_test_set[poly_degree-1, 1] = variance_test_set
    

plot_bias_and_variance_test_set(poly_degree_biases_test_set, poly_degree_variances_test_set)


    