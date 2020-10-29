# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:09:51 2020

@author: rubenandrebarreiro
"""

# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import SciKit-Learn as skl
import sklearn as skl

# Import Support Vector Classifier,
# from the SciKit-Learn's Support Vector Machine Module,
# as support_vector_machine
from sklearn.svm import SVC as support_vector_machine

# Import the Support Vector Machine Plot Functions,
# from the Customised T4_Aux Python's Library
from files.T4aux import plot_svm as plot_support_vector_machine

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


c_param_initial_value = 1e-1

for c_param_factor in range(3):
    
    c_param_value = ( c_param_initial_value * 10**c_param_factor )
    
    print("-------------------------------------------------------------")
    print("\n")
    print("Trying the Regularization Parameter C = {}...".format(c_param_value))
    print("\n")
    
    
    support_vector_machine_classifier = support_vector_machine(kernel='linear', C=c_param_value)
    
    support_vector_machine_classifier.fit(xs_dataset_features_std, ys_dataset_classes)
    
    
    plot_support_vector_machine_image_filename = "imgs/support-vector-machine-plot-c-{}.png".format(c_param_value)
    
    
    plot_support_vector_machine(dataset_random, support_vector_machine_classifier, plot_support_vector_machine_image_filename, c_param_value)

    
    support_vectors_indexes = support_vector_machine_classifier.support_
       
    support_vector_dual_coefs = support_vector_machine_classifier.dual_coef_
    
    
    print("Support Vectors' Indexes, with C = {}:".format(c_param_value))
    print("\n")
    print(support_vectors_indexes)
    print("\n\n")
    
    print("Support Vectors' Dual Coefficients, with C = {}:".format(c_param_value))
    print("\n")
    print(support_vector_dual_coefs)
    print("\n\n")
    
    
    # ------------------------------------------------- #  
    # Question 1 - How many support vectors are there   #
    #              for each class (-1 and 1),           #
    #              for each value of C?                 #
    # ------------------------------------------------- #  
    
    support_vector_dual_coefs_positive_class = support_vector_dual_coefs[support_vector_dual_coefs >= 0]
    
    support_vector_dual_coefs_negative_class = support_vector_dual_coefs[support_vector_dual_coefs <= 0]
    
    
    print("Support Vectors' Dual Coefficients, with C = {}, for Class 1:".format(c_param_value))
    print("\n")
    print(support_vector_dual_coefs_positive_class)
    print("\n\n")
    
    print("Support Vectors' Dual Coefficients, with C = {}, for Class -1:".format(c_param_value))
    print("\n")
    print(support_vector_dual_coefs_negative_class)
    print("\n\n")


    num_samples_support_vector_dual_coefs_positive_class = len(support_vector_dual_coefs_positive_class)    
    num_samples_support_vector_dual_coefs_negative_class = len(support_vector_dual_coefs_negative_class)
   
     
    print("\n")
    print("#########################################################################")
    print("\n")    
    
    print("# Question 1 - How many support vectors are there for each class (-1 and 1), for each value of C?")
    
    print("Number of Samples/Examples, with C = {}, which belongs to Class 1: {}".format(c_param_value, num_samples_support_vector_dual_coefs_positive_class))
    print("Number of Samples/Examples, with C = {}, which belongs to Class -1: {}".format(c_param_value, num_samples_support_vector_dual_coefs_negative_class))
        
    print("\n")
    print("#########################################################################")
    print("\n")    
    
    
    # ----------------------------------------------------------- #  
    # Question 2 - How many of those Support Vectors correspond   #
    #              to examples at the margin or examples that     #
    #              violate the margin constraint?                 #
    # ----------------------------------------------------------- #  
    
    support_vector_dual_coefs_violate_restriction = support_vector_dual_coefs[abs(support_vector_dual_coefs) == c_param_value]
    
    support_vector_dual_coefs_not_violate_restriction = support_vector_dual_coefs[abs(support_vector_dual_coefs) != c_param_value]
    
    
    print("Support Vectors' Dual Coefficients, with C = {}, which VIOLATE the Restrictions (i.e., ARE INSIDE the margins):".format(c_param_value))
    print("\n")
    print(support_vector_dual_coefs_violate_restriction)
    print("\n\n")
    
    print("Support Vectors' Dual Coefficients, with C = {}, which DON'T VIOLATE the Restrictions (i.e., AREN'T INSIDE the margins):".format(c_param_value))
    print("\n")
    print(support_vector_dual_coefs_not_violate_restriction)
    print("\n\n")


    num_samples_support_vector_dual_coefs_violate_restriction = len(support_vector_dual_coefs_violate_restriction)    
    num_samples_support_vector_dual_coefs_not_violate_restriction = len(support_vector_dual_coefs_not_violate_restriction)
   
    
    print("\n")
    print("#########################################################################")
    print("\n")    
    
    print("# Question 2 - How many of those Support Vectors correspond to examples at the margin or examples that violate the margin constraint?")
    
    print("Number of Support Vectors, with C = {}, which VIOLATE the Restrictions (i.e., ARE INSIDE the margins): {}".format(c_param_value, num_samples_support_vector_dual_coefs_violate_restriction))
    print("Number of Support Vectors, with C = {}, which DON'T VIOLATE the Restrictions (i.e., AREN'T INSIDE the margins): {}".format(c_param_value, num_samples_support_vector_dual_coefs_not_violate_restriction))
        
    print("\n")
    print("#########################################################################")
    print("\n")    
    
    
    # ----------------------------------------------------------- #  
    # Question 3 - Explain the variation of the margins with      #
    #              the value of C.                                #
    # ----------------------------------------------------------- #  
    # Answer 3 - The Variation of the C Regularization Paramater, #
    #            changes the behaviour of the margins,            #
    #            making them more soft or more strict             #
    #            i.e., with smaller C Regularization Parameters,  #
    #            we got soft margins constraints and,             #
    #            with higher C Regularization Parameters,         #
    #            we got strict margins constraints                #
    # ----------------------------------------------------------- #

    
    
    # ----------------------------------------------------------- #  
    # Question 4 - Explain the variation in                       #
    #              the number of support vectors                  #
    #              that violate the margin constraint as you      #
    #              vary the C value.                              #
    # ----------------------------------------------------------- #  
    # Answer 4 - As we increase the number of                     #
    #            C Regularization Parameter,                      #
    #            the number of Support Vectors                    #
    #            which violate the margin constraints decrease.   #
    #            In other words,                                  #
    #            with smaller C Regularization Parameters,        #
    #            we got more Support Vectors                      #
    #            violating the margins constraints and,           #
    #            with higher C Regularization Parameters,         #
    #            we got less Support Vectors                      #
    #            violating the margins constraints.               #
    # ----------------------------------------------------------- #
