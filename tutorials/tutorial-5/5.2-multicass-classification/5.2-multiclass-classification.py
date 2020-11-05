# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:32:06 2020

@author: rubenandrebarreiro
"""

# Import NumPy Python's Library as np
import numpy as np

# Import Logistic Regression Sub-Module, from SciKit-Learn Python's Library,
# as skl_logistic_regression 
from sklearn.linear_model import LogisticRegression as skl_logistic_regression

# Import the Logistic Regression Classifier Plot Function,
# from the Customised T5_Plot Python's Library
from files.T5plot import plot_logregs as plot_logistic_regressions

# Import the Support One vs. Rest Classifier Plot Function,
# from the Customised T5_Plot Python's Library
from files.T5plot import plot_ovr as plot_one_vs_rest

# From SciKit-Learn, import Datasets as skl_datasets
from sklearn import datasets as skl_datasets

# Import OneVsRestClassifier Sub-Module, from SciKit-Learn.Multiclass Python's Library,
# as skl_one_vs_rest_classifier
from sklearn.multiclass import OneVsRestClassifier as skl_one_vs_rest_classifier

# Import SVC Sub-Module, from SciKit-Learn.SVC Python's Library,
# as support_vector_classifier
from sklearn.svm import SVC as support_vector_classifier


# Load the Iris' Dataset, from the SciKit-Learn's Datasets
iris_dataset = skl_datasets.load_iris()


# Set the features of the Iris' Dataset
xs_features = iris_dataset.data[:,[0,1]]

# Set the classes of the Iris' Dataset
ys_classes = iris_dataset.target

# The number of Samples
num_samples = len(xs_features)

# The list of Logistic Regression Classifiers
logRegs_classifiers = []

# The C Regularization Parameter
c_param_value = 1e1000


# Loop to iterate all the Iris' Classes
for ys_class_iris in range(3):
    
    # Create an array of Classes,
    # for the current Logistic Regression Classifier
    ys_classes_current_logReg = np.zeros(num_samples)
    
    # Set the new Classes, based on the current iterated Iris' Class
    ys_classes_current_logReg[ys_classes == ys_class_iris] = 0
    ys_classes_current_logReg[ys_classes != ys_class_iris] = 1
   
    
    # Initialise the current Logistic Regression Classifier,
    # for the current Settings
    logReg = skl_logistic_regression(C=c_param_value, tol=1e-10)              
    
    # Fit the Logistic Regression Classifier with the Training Set
    logReg.fit(xs_features, ys_classes_current_logReg)      

    # Append the current Logistic Regression Classifier,
    # for the current Settings
    logRegs_classifiers.append(logReg)                     


# Plot a customized One vs. Rest Algorithm, for the Logistic Regression Classifier,
# in the classification of the Iris' Dataset problem
plot_logistic_regressions(logRegs_classifiers, "one-vs-rest-logistic-regression-iris-dataset.png")

# Initialise the One vs. Rest Algorithm,
# from the SciKit-Learn's Python Library, for Support Vector Machines,
# with a Kernel of Gaussian RBF, a Gamma of 0.7 and a C Regularization Parameter of 10
svmOneVsRest = skl_one_vs_rest_classifier(support_vector_classifier(kernel='rbf', gamma=0.7, C=10))

# Fit the One vs. Rest Algorithm, from the SciKit-Learn's Python Library,
# for Support Vector Machines, with the Training Set
svmOneVsRest.fit(xs_features, ys_classes)

# Plot the One vs. Rest Algorithm, from the SciKit-Learn's Python Library,
# for the Support Vector Machine Classifier,
# in the classification of the Iris' Dataset problem
plot_one_vs_rest(svmOneVsRest, "one-vs-rest-support-vector-machines-iris-dataset.png")













