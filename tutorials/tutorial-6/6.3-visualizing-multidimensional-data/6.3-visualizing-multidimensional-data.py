# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:56:32 2020

@author: rubenandrebarreiro
"""

# Import PyPlot Sub-Module, from Matplotlib Python's Library as plt
import matplotlib.pyplot as plt

# Import read_csv Sub-Module,
# from Pandas Python's Library
from pandas import read_csv

# Import plotting.scatter_matrix Sub-Module,
# from Pandas Python's Library
from pandas.plotting import scatter_matrix

# Import plotting.parallel_coordinates Sub-Module,
# from Pandas Python's Library
from pandas.plotting import parallel_coordinates

# Import plotting.andrews_curves Sub-Module,
# from Pandas Python's Library
from pandas.plotting import andrews_curves


# The file of the Iris' Data
iris_data_file = "../files/data/iris.data"


# Set the Style of the Plots, as 'Seaborn' Style
plt.style.use('seaborn')

# The Iris' Data, read from the file
iris_data = read_csv(iris_data_file)


# The Function to plot the Stacked Histograms
def plot_stacked_histograms(iris_data, alpha_value):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Stacked Histograms
    iris_data.plot(kind='hist', bins=15, alpha=alpha_value)
        
    # Set the label for the X axis of the Plot
    plt.xlabel("Individual Features")
    
    # Set the label for the Y axis of the Plot
    plt.ylabel("Frequency")
    
    # Set the Title of the Plot
    plt.title('Frequency of Individual Features\nfor the Iris\' Dataset, with alpha={}'.format(alpha_value))
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-stacked-histograms-alpha-{}.png'.format(alpha_value), dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


# The Function to plot the Individual Histograms
def plot_individual_histograms(iris_data, alpha_value):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Individual Histograms
    iris_data.hist(bins=15, alpha=alpha_value)
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-individual-histograms-alpha-{}.png'.format(alpha_value), dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
    
# The Function to plot the Box
def plot_box(iris_data):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Box
    iris_data.plot(kind='box')
    
    # Set the Title of the Plot
    plt.title('Box Plot')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-box.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
    
# The Function to plot the Scatter Matrix, with KDE Diagonal
def plot_scatter_matrix_kde_diagonal(iris_data, alpha_value):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Scatter Matrix, with KDE Diagonal
    scatter_matrix(iris_data, alpha=alpha_value, figsize=(15,10), diagonal='kde')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-scatter-matrix-kde-diagonal-alpha-{}.png'.format(alpha_value), dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    
    
# The Function to plot the Scatter Matrix, with Histogram Diagonal
def plot_scatter_matrix_histogram_diagonal(iris_data, alpha_value):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Scatter Matrix, with Histogram Diagonal
    scatter_matrix(iris_data, alpha=alpha_value, figsize=(15,10), diagonal='hist')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-scatter-matrix-histogram-diagonal-alpha-{}.png'.format(alpha_value), dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


# The Function to plot the Parallel Coordinates
def plot_parallel_coordinates(iris_data):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Parallel Coordinates
    parallel_coordinates(iris_data, 'Name')
    
    # Set the Title of the Plot
    plt.title('Parallel Coordinates,\nrepresenting the different Classes in different Colours')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-parallel-coordinates.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()
    

# The Function to plot the Andrew's Curves
def plot_andrews_curves(iris_data):
    
    # Initialise the Plot
    plt.figure(figsize=(10, 8), frameon=True)

    # Plot the Parallel Coordinates
    andrews_curves(iris_data, 'Name')
    
    # Set the Title of the Plot
    plt.title('Andrew\'s Curves,\nrepresenting the different Classes in different Colours')
    
    # Save the Plot, as a figure/image
    plt.savefig('imgs/iris-data-andrew-curves.png', dpi=600)
    
    # Show the Plot
    plt.show()
    
    # Close the Plot
    plt.close()


# Plot several Stacked Histograms, varying the Alpha value
plot_stacked_histograms(iris_data, 0.25)    
plot_stacked_histograms(iris_data, 0.5)    
plot_stacked_histograms(iris_data, 0.75)
plot_stacked_histograms(iris_data, 1.0)


# Plot several Individual Histograms, varying the Alpha value
plot_individual_histograms(iris_data, 0.25)    
plot_individual_histograms(iris_data, 0.5)    
plot_individual_histograms(iris_data, 0.75)
plot_individual_histograms(iris_data, 1.0)


# Plot Boxes
plot_box(iris_data)


# Plot several Scatter Matrices, with KDE Diagonal,
# varying the Alpha value
plot_scatter_matrix_kde_diagonal(iris_data, 0.25)    
plot_scatter_matrix_kde_diagonal(iris_data, 0.5)    
plot_scatter_matrix_kde_diagonal(iris_data, 0.75)
plot_scatter_matrix_kde_diagonal(iris_data, 1.0)


# Plot several Scatter Matrices, with Histogram Diagonal,
# varying the Alpha value
plot_scatter_matrix_histogram_diagonal(iris_data, 0.25)    
plot_scatter_matrix_histogram_diagonal(iris_data, 0.5)    
plot_scatter_matrix_histogram_diagonal(iris_data, 0.75)
plot_scatter_matrix_histogram_diagonal(iris_data, 1.0)


# Plot Parallel Coordinates
plot_parallel_coordinates(iris_data)


# Plot Andrew's Curves
plot_andrews_curves(iris_data)