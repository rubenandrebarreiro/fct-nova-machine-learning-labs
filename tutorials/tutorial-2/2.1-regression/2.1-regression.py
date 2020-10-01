# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:19:10 2020

@author: Ruben Andre Barreiro
"""

# Import NumPy Python's Library
import numpy as np


# Load Bluegill Fishes' data from the .txt file,
# with delimeter '\t' and skipping the first row
# (the labels of the data)
# NOTE: 1) The 1st column corresponds to
#          the different ages of the fishes (in years)
#       2) The 2nd column corresponds to
#          the size of the fishes (in millimeters)
bluegill_fishes_data = np.loadtxt('../files/bluegills.txt', delimiter='\t', skiprows=1)


# Adjust the Scale of the data, to the maximum of
# the 'x' values ( x = age (in Years) )
scale = np.max(bluegill_fishes_data, axis = 0)

# Adjust the scale of the Bluegill Fishes' data
bluegill_fishes_data = bluegill_fishes_data / scale



# Print the Bluegill Fishes' data
# NOTE: 1) The 1st column corresponds to
#          the different ages of the Bluegill Fishes (in years)
#       2) The 2nd column corresponds to
#          the size of the Bluegill Fishes (in millimeters)
print('\n')
print('The Bluegill Fishes\' data ( x = age (in Years) ; y = length (in Millimeters) )')
print(bluegill_fishes_data)



# The Bluegill Fishes' 'x's, containing the Fishes' Ages (in Years) 
bluegill_fishes_data_xs_age_in_years = bluegill_fishes_data[:, :-1]

# The Bluegill Fishes' 'y's, containing the Fishes' Sizes (in Millimeters) 
bluegill_fishes_data_ys_size_in_millimeters = bluegill_fishes_data[:, [-1]]

# The total number of rows of
# the Bluegill Fishes' 'x's, containing the Fishes' Ages (in Years)
total_num_rows_bluegill_fishes_data = len(bluegill_fishes_data_xs_age_in_years)


# Print the Bluegill Fishes' 'x's, containing the Fishes' Ages (in Years)
print('\n')
print('The Bluegill Fishes\' data ( x = age (in Years) )')
print(bluegill_fishes_data_xs_age_in_years)


# Print the Bluegill Fishes' 'y's, containing the Fishes' Sizes (in Millimeters)
print('\n')
print('The Bluegill Fishes\' data ( y = length (in Millimeters) )')
print(bluegill_fishes_data_ys_size_in_millimeters)



# The Mean (Average) of the Bluegill Fishes' data,
# about the X's (Age, in Years) of the Fishes
bluegill_fishes_data_mean_average = np.mean(bluegill_fishes_data_xs_age_in_years, axis=0)

# The Standard Deviation of the Bluegill Fishes' data,
# about the X's (Age, in Years) of the Fishes
bluegill_fishes_data_standard_deviation = np.std(bluegill_fishes_data_xs_age_in_years, axis=0)


# Print the Mean (Average) of the Bluegill Fishes' data,
# about the X's (Age, in Years) of the Fishes
print('\n')
print('The Bluegill Fishes\' data Mean/Average ( for x = age (in Years) )')
print(bluegill_fishes_data_mean_average)


# Print the Standard Deviation of the Bluegill Fishes' data,
# about the X's (Age, in Years) of the Fishes
print('\n')
print('The Bluegill Fishes\' data Standard Deviation ( for x = age (in Years) )')
print(bluegill_fishes_data_standard_deviation)


# The Standardization of the Bluegill Fishes' data
bluegill_fishes_data_standardization = ( bluegill_fishes_data - bluegill_fishes_data_mean_average ) / bluegill_fishes_data_standard_deviation



# Print the Bluegill Fishes' data standardized
print('\n')
print('The Bluegill Fishes\' data standardized ( for x = age (in Years) ):')
print(bluegill_fishes_data_standardization)


# Return 2 Matrices, splitting the data, at random
def random_split(data,test_points):

    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)

    train = data[ranks>=test_points,:]
    test = data[ranks<test_points,:]

    return train,test


# The Standardization of the Bluegill Fishes' data, randomized
bluegill_fishes_data_standardization_random = np.copy(bluegill_fishes_data_standardization)


# Print the Bluegill Fishes' data standardized
print('\n')
print('The Bluegill Fishes\' data standardized and randomised ( for x = age (in Years) ):')
print(bluegill_fishes_data_standardization_random)


# Split the Bluegill Fishes' data, in 3 subsets:
# 1) The Bluegill Fishes' data for Training (0.5 of the Global Set)
# 2) The Bluegill Fishes' data for Validation (0.25 of the Global Set)
# 3) The Bluegill Fishes' data for Testing (0.25 of the Global Set)
bluegill_fishes_data_training, temp = random_split(bluegill_fishes_data_standardization_random, int(bluegill_fishes_data.shape[0]/2))
bluegill_fishes_data_validation, bluegill_fishes_data_testing = random_split(temp, int(temp.shape[0]/2))

# Print the Bluegill Fishes' data for Training (0.5 of the Global Set)
print('\n')
print('The Bluegill Fishes\' data for Training (0.5 of the Global Set):')
print(bluegill_fishes_data_training)


# Print the Bluegill Fishes' data for Validation (0.25 of the Global Set)
print('\n')
print('The Bluegill Fishes\' data for Validation (0.25 of the Global Set):')
print(bluegill_fishes_data_validation)


# Print the Bluegill Fishes' data for Testing (0.25 of the Global Set)
print('\n')
print('The Bluegill Fishes\' data for Testing (0.25 of the Global Set):')
print(bluegill_fishes_data_testing)


# Import the Warning's Python's Library
import warnings


# Ignore the Rank Warnings, for the PolyFit function,
# in the case of the amount of data, leads to unreliable fitting
warnings.simplefilter('ignore', np.RankWarning)



# Import the Matplotlib Python's Library
import matplotlib.pyplot as plt


# Function to plot the several Sets for the Models:
# 1) Training Set
# 2) Validation Set
# 3) Testing Set
def plotSets(training_set, validation_set, testing_set):
    plt.plot(training_set[:,0], training_set[:,1],'or')
    plt.plot(validation_set[:,0], validation_set[:,1],'og')
    plt.plot(testing_set[:,0], testing_set[:,1],'ob')


# Create a Figure, for Plotting
plt.figure(1, figsize=(12, 8), frameon=True)

# Set a title for the Plotting Figure
plt.title('Bluegill Fishes\' Size')

# Plot the several Sets for the Models:
# 1) Training Set
# 2) Validation Set
# 3) Testing Set
plotSets(bluegill_fishes_data_training, bluegill_fishes_data_validation, bluegill_fishes_data_testing)


# Creates a Linear Space, with 100 points,
# from the minimum value to the maximum value of the data standardized ,
# for the parameters (theta vector)
pxs = np.linspace(min(bluegill_fishes_data_standardization[:,0]),max(bluegill_fishes_data_standardization[:,0]),100)


# Return the Mean Squared Error
# NOTE: X on 1st column ( for x = age (in Years) )
#       Y on 2nd column ( for y = length (in Millimeters) )
def mean_squared_error(data,coefficients):
    
    pred_ys = np.polyval(coefficients, data[:,0])
    mean_squared_error = np.mean((data[:,1]-pred_ys)**2)
    
    return mean_squared_error



# Very large number, for the Best Validation Error
best_validation_err = 10000000


# For the several degrees from 1 to 6,
# fit the Polynomial Curve and plot each one, as also,
# compute the Training and Validation Errors
for degree in range(1,7):

    # Compute the vector 
    coefficients = np.polyfit(bluegill_fishes_data_training[:,0], bluegill_fishes_data_training[:,1], degree)
    
    # Compute the Training error, from the Bluegill Fishes' data,
    # based on the Mean Squared Error
    training_error = mean_squared_error(bluegill_fishes_data_training, coefficients)

    # Compute the Validation error, from the Bluegill Fishes' data,
    # based on the Mean Squared Error
    validation_error = mean_squared_error(bluegill_fishes_data_validation, coefficients)
    
    
    # Compute several metrics:
    # 1) The Best Validation Error
    # 2) The Best Validation Coefficient
    # 3) The Best Validation Degree
    if validation_error < best_validation_err:
        best_validation_error = validation_error
        best_validation_coefficient = np.copy(coefficients)
        best_validation_degree = degree
    
    
    # Create the Label for the Plotting,
    # for the several Training and Validation Errors
    label_plot = '{g:d}/{t:6.3f}/{v:6.3f}'.format(g=degree, t=training_error, v=validation_error)
    
    # The 'y's predicted from the Polynomial Curve
    pred_ys = np.polyval(coefficients, pxs)
    
    # Plot the Polynomial Curve for the several 'y's predicted
    plt.plot(pxs, pred_ys, '-', label=label_plot)
    
    # Set the Legend for the Plotting,
    # for the previously defined labels
    plt.legend(loc='center right', borderaxespad=0.8)
    

# Save a figure for the Plotting of the Bluegill Fishes'
# predicted data points
# NOTE: This needs to be done always after the plot function
#       and before the close function
plt.savefig('../files/imgs/bluegill-data-1.png',dpi=300)

# Show the Plot, containing
# the points of the Bluegill Fishes' data and
# the several Fitted Degrees Polynomial Function
plt.show()

# Close the Figure for Plotting
plt.close()


# Compute the Testing error, from the Bluegill Fishes' data,
# based on the Mean Squared Error
testing_error = mean_squared_error(bluegill_fishes_data_testing, best_validation_coefficient)


# Print the Best Degree for Validation
print('\n')
print('Best Degree for Validation:')
print(best_validation_degree)


# Print the Testing Error of the Best Hypothesis (with the Best Degree)
print('\n')
print('Testing Error of the Best Hypothesis (with the Best Degree):')
print(testing_error)


