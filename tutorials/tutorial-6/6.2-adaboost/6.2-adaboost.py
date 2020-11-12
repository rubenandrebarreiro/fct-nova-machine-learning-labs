# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:34:28 2020

@author: rubenandrebarreiro
"""

# Definition of the necessary Python Libraries

# a) General Libraries:

# Import NumPy Python's Library as np
import numpy as np

# Import SciKit-Learn as skl
import sklearn as skl

# Import Tree.DecisionTreeClassifier Sub-Module,
# from SciKit-Learn Python's Library as decision_tree_classifier
from sklearn.tree import DecisionTreeClassifier as decision_tree_classifier

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import System Python's Library
import sys

# Append the Path "../" to the System's Path
sys.path.append('../')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The file of the Dataset
dataset_file = "../files/data/T6-data.txt"

# Load the Data of the Dataset with NumPy function loadtxt
dataset_not_random = np.loadtxt(dataset_file, delimiter="\t")    

# Shuffle the Dataset, not randomized
dataset_random = skl.utils.shuffle(dataset_not_random)

# Select the Classes of the Dataset, randomized
ys_classes = dataset_random[:, -1]

# Select the Features of the Dataset, randomized
xs_features = dataset_random[:, 0:-1]     

# The size of the Dataset, randomized
dataset_size = len(xs_features)

# Compute the Means of the Dataset, randomized
dataset_means = np.mean(xs_features, axis=0)

# Compute the Standard Deviations of the Dataset, randomized
dataset_stdevs = np.std(xs_features, axis=0)

# Standardize the Dataset, randomized
xs_features_std = ( ( xs_features - dataset_means ) / dataset_stdevs )

# Update the Classes of the Dataset,
# to have the value of 1 or -1
ys_classes = ( ( ys_classes * 2 ) - 1 )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The Decision Stumps of the Hypothesis for the Models, for Boosting
hypothesis_decision_stumps = []

# The Weights of the Hypothesis for the Models, for Boosting
hypothesis_weights = []

# The Weights for the Points, initialized with the same Weight of 1/N,
# for all the Data Points, where N is the number of Points in the Dataset
point_weights = ( np.ones(xs_features_std.shape[0]) / float(xs_features_std.shape[0]) )


# The maximum number of Hypothesis, for Boosting
max_hypothesis = 20

# The Number of Prediction Errors, for each Hypohtesis Model/Cycle
boosting_errors_predictions = np.zeros((20, 2))

# Iterate the Loop for a number of a maximum of 20 Hypothesis, for Boosting
for ix in range(max_hypothesis):
    
    # Initialise the Decision Stump, from the Decision Tree Classifier,
    # with a Maximum Depth, equal to 1
    decision_tree_classifier_stump = decision_tree_classifier(max_depth=1)
    
    # Fit the Decision Tree, with the Features and Classes of the Dataset,
    # as also, with the Weights pre-computed previously
    decision_tree_classifier_stump.fit(xs_features_std, ys_classes, sample_weight = point_weights)
    
    # Predict the Classes, for the Features of the Dataset,
    # using the current Hypothesis
    pred = decision_tree_classifier_stump.predict(xs_features_std)
    
    
    # Compute the Errors of the Predictions, against the Real Classes,
    # y_m(x^n) != t^n
    errors_pred = (pred != ys_classes).astype(int)
    
    # Compute the Sum of the Errors Weighted 
    error_sum_weighted = np.sum(errors_pred * point_weights)
    
    
    # Compute the Alpha value, alpha_m = ln( (1-e_m) / e_m )
    alpha = np.log( (1 - error_sum_weighted) / error_sum_weighted )
    
    
    # Update the Weigths of the Points, for the next loop cycle,
    # w[^n;_m+1] = w[^n;_m] * exp( alpha_m * I( y_m(x^n) != t^n ) )
    point_weights = ( point_weights * np.exp(alpha * errors_pred) )
    
    # Normalize the Weights, after computing new Weights of the Points,
    # dividing them by the sum of the Weights of the Points
    point_weights = ( point_weights / np.sum(point_weights) )
    
    
    # Append the Decision Stump, for the current Hypothesis,
    # to the Decision Stumps of the Hypothesis for the Models, for Boosting
    hypothesis_decision_stumps.append(decision_tree_classifier_stump)
    
    # Append the Alpha value,
    # to the Weights of the Hypothesis for the Models, for Boosting
    hypothesis_weights.append(alpha)
    
    
    preds = np.zeros(dataset_size)
    weighted_preds = np.zeros(dataset_size)
    
    for ix_final_prediction in range(len(hypothesis_decision_stumps)):
        preds = hypothesis_decision_stumps[ix_final_prediction].predict(xs_features_std)
        weighted_preds = ( weighted_preds + ( preds * hypothesis_weights[ix_final_prediction] ) )
    
    weighted_preds[weighted_preds < 0] = -1
    weighted_preds[weighted_preds >= 0] = 1
    
    boosting_errors_predictions[ix, 0] = ( ix + 1 )
    boosting_errors_predictions[ix, 1] = np.sum((weighted_preds != ys_classes).astype(int))
    

# The Function to plot the Prediction Errors, for AdaBoost
def plot_prediction_errors(boosting_pred_errors):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Set the line representing the continuous values,
    # for the Functions of the Single/Ensemble Bagging Training and Validation Errors
    plt.plot(boosting_pred_errors[:,0], boosting_pred_errors[:,1], '-', color="blue")
    
    # Set the axis for the Plot
    plt.axis([0, 21, 0.0, max(boosting_pred_errors[:,1])])
    
    # Set the laber for the X axis of the Plot
    plt.xlabel("Number of Hypothesis/Cycles")
    
    # Set the laber for the Y axis of the Plot
    plt.ylabel("Prediction Errors, for AdaBoost Ensemble Method")
    
    # Set the Title of the Plot
    plt.title('Prediction Errors, for AdaBoost Ensemble Method,\nwith 20 Hypothesis/Cycles')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/adaboost-prediction-errors.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
print("\nPerforming the Adaboost (Adaptative Boost Process)...\n\n")
print("Number of Errors of the Predictions for the Adaboost Ensemble Method,\nfor all the Hypothesis/Cycles:")
print(boosting_errors_predictions)
print("\nConclusion:\n- Even with a Bad Classifier, using the Adaboost,\n  we ensure a minimization of the number of\n  the Errors of the Predictions...\n\n")

plot_prediction_errors(boosting_errors_predictions)
