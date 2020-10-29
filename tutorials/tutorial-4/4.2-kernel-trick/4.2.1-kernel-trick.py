# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:40:21 2020

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

# Import Support Vector Classifier,
# from the SciKit-Learn's Support Vector Machine Module,
# as support_vector_machine
from sklearn.svm import SVC as support_vector_machine

# Import the Support Vector Machine Plot Functions,
# from the Customised T4_Aux Python's Library
from files.T4aux import plot_svm_mark_wrong_x as plot_support_vector_machine_corrects_o_wrongs_x

# Import System Python's Library
import sys

# Append the Path "../" to the System's Path
sys.path.append('../')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

for current_support_vector_machine_kernel_index in range(len(support_vector_machine_kernels)):
    
    current_support_vector_machine_kernel = support_vector_machine_kernels[current_support_vector_machine_kernel_index]
    
    print("-------------------------------------------------------------")
    print("\n")
    print("Trying the \"{}\" Kernel, to the Support Vector Machine Classifier, with the following Parameters:".format(current_support_vector_machine_kernel[0]))
    print("- C Regularization Parameter: 1.0 ;")
    print("- Degree: {} (only considered in \"Poly\" Kernel) ;".format(current_support_vector_machine_kernel[1]))
    print("- Gamma: {} ;".format(current_support_vector_machine_kernel[2]))
    print("- r value (coef_0 parameter): {} (only considered in \"Poly\" and \"Sigmoid\" Kernels) ;".format(current_support_vector_machine_kernel[3]))
    print("\n")
    
    
    support_vector_machine_classifier = support_vector_machine(kernel=current_support_vector_machine_kernel[0], degree=current_support_vector_machine_kernel[1], gamma=current_support_vector_machine_kernel[2], coef0=current_support_vector_machine_kernel[3], C=1.0)
    
    support_vector_machine_classifier.fit(xs_train_features_std, ys_train_classes)
    
    
    plot_support_vector_machine_image_filename = "imgs/support-vector-machine-plot-c-1.0-{}-kernel.png".format(current_support_vector_machine_kernel[0])
    
    
    # Prepare Data from the Training Set, for the Plot Function
    
    # Create a Matrix, with num_samples_train rows,
    # and 3 columns (2 features + 1 class)
    dataset_train = np.zeros((len(xs_train_features_std), 3))
    
    # Fill the Data from the Training Set, for the Plot Function
    dataset_train[:,0:2] = xs_train_features_std[:,:]
    dataset_train[:,2] = ys_train_classes
    
    plot_support_vector_machine_corrects_o_wrongs_x(dataset_train, support_vector_machine_classifier, current_support_vector_machine_kernel[0], plot_support_vector_machine_image_filename, 1.0)

    
    support_vectors_indexes = support_vector_machine_classifier.support_
       
    support_vector_dual_coefs = support_vector_machine_classifier.dual_coef_
    
    