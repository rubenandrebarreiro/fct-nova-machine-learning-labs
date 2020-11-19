# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:00:36 2020

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

# Import PCA,
# From the Decomposition Module,
# of the SciKit-Learn's Python Library,
# as pca_decomposition
from sklearn.decomposition import PCA as pca_decomposition

# Load the Iris' Dataset
iris_dataset = load_iris_dataset()

# Select the Features of the Iris' Dataset
xs_features = iris_dataset.data

# Select the Classes of the Iris' Dataset
ys_classes = iris_dataset.target


# Set the PCA Decomposition, for Feature Extraction
pca = pca_decomposition(n_components=2)

# Fit the PCA Decomposition with Features
pca.fit(xs_features)

print("PCA Components:")
print(pca.components_)

print("\n\n")

# Transformed Data,
# from the PCA Decomposition Transform function
# on the Features
transformed_xs_features_data = pca.transform(xs_features)

print("Transformed Features' Data:")
print(transformed_xs_features_data)

print("\n\n")


def plot_iris(X,y,file_name):
    plt.figure(figsize=(7,7))
    
    # NOTE:
    # - Invert the Ys, to make a reflection, regarding the X-Axis,
    #   for a better presentation of the plot
    plt.plot(X[y==0,0], -X[y==0,1],'o', markersize=7, color='blue', alpha=0.5)
    plt.plot(X[y==1,0], -X[y==1,1],'o', markersize=7, color='red', alpha=0.5)
    plt.plot(X[y==2,0], -X[y==2,1],'o', markersize=7, color='green', alpha=0.5)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set the Title of the Plot
    plt.title('Feature Extraction with PCA (with 2 Components)')
    
    # Save the Plot
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
# Plot the Iris' New Features' Values,
# from Feature Extraction with PCA
plot_iris(transformed_xs_features_data, ys_classes, "imgs/feature-extraction-with-pca-2-components.png")