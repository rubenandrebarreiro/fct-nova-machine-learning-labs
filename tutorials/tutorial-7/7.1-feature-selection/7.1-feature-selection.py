# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:08:38 2020

@author: rubenandrebarreiro
"""

# Import PyPlot Sub-Module,
# From Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import Load Iris,
# From the Datasets Module,
# of the SciKit-Learn's Python Library,
# as load_iris_dataset
from sklearn.datasets import load_iris as load_iris_dataset

# Import Function Classifier,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as function_classifier
from sklearn.feature_selection import f_classif as f_classifier

# Import Select K Best,
# From the Feature Selection Module
# of the SciKit-Learn's Python Library,
# as select_k_best_features
from sklearn.feature_selection import SelectKBest as select_k_best_features

# Load the Iris' Dataset
iris_dataset = load_iris_dataset()

# Select the Features of the Iris' Dataset
xs_features = iris_dataset.data

# Select the Classes of the Iris' Dataset
ys_classes = iris_dataset.target

# Select the F-Values and the Probabilities, from the F-Test
f_values, probabilities_f_values = f_classifier(xs_features, ys_classes)

print("F-Values:")
print(f_values)

print("\n")

print("Probabilities for F-Values:")
print(probabilities_f_values)

print("\n")

best_features = []

for current_best_feature_selection in range(2):
    
    best_f_value = -99999999999
    best_feature = -1
    
    for current_feature in range(len(f_values)):
        if( (current_feature not in best_features) and (f_values[current_feature] > best_f_value) ):
            best_f_value = f_values[current_feature]
            best_feature = current_feature
            
    best_features.append(best_feature)

print("Best Features Indexes (manually selected):")
print(best_features)

print("\n")

xs_new_features_1 = xs_features[:,best_features]

xs_new_features_2 = select_k_best_features(f_classifier, k=2).fit_transform(xs_features, ys_classes)

print("Best Features (manually selected):")
print(xs_new_features_1)

print("\n")
            
print("Best Features (selected by the Select K Best Features):")
print(xs_new_features_2)

print("\n")


def plot_iris(X,y,file_name):
    plt.figure(figsize=(7,7))
    
    plt.plot(X[y==0,0], X[y==0,1],'o', markersize=7, color='blue', alpha=0.5)
    plt.plot(X[y==1,0], X[y==1,1],'o', markersize=7, color='red', alpha=0.5)
    plt.plot(X[y==2,0], X[y==2,1],'o', markersize=7, color='green', alpha=0.5)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set the Title of the Plot
    plt.title('Selection 2 Best Features (F-Test)')
    
    # Save the Plot
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
    
# Plot the Iris' New Features' Values,
# from Feature Selection 2 Best Features
plot_iris(xs_new_features_1, ys_classes, "imgs/selection-best-2-features.png")