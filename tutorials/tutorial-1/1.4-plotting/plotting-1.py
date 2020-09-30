# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:50:14 2020

@author: Ruben Andre Barreiro
"""

# Import NumPy Python's Library
import numpy

# Return Matrix with Planets' Orbital Radius and Orbital Periods
# Note: Each row corresponds to a Planet:
#       - The 1st Column corresponds to the Orbital Radius
#       - The 2nd Column corresponds to the Orbital Period
def load_planet_data(file_name):
    
    rows = []
 
    lines = open(file_name).readlines()
 
    for line in lines[1:]:
        parts = line.split(',')
        rows.append( (float(parts[1]),float(parts[2])) )

    return numpy.array(rows)

planets_data = load_planet_data('../files/planets.csv')

# Import Matplotlib Python's Library
import matplotlib.pyplot as plt

# Enable the Matplotlib command, in line,
# for IPython's interactive console
# Note: Uncomment, if you want
#%matplotlib inline

# Create a Figure for Plotting
plt.figure()

# Plot the points of the Planets' data
plt.plot(planets_data[:,0],planets_data[:,1])

# Save a figure for the Plotting of the Planets' data points
# NOTE: This needs to be done always after the plot function
#       and before the close function
plt.savefig('../files/imgs/planets-data-1.png',dpi=300)

# Show the Plot, containing
# the points of the Planets' data
plt.show()

# Close the Figure for Plotting
plt.close()


# Extra - Plotting the Planets' data points,
#         with other configuration
#         (plotting 'x's, instead of a straight line)

# Create a Figure for Plotting
plt.figure()

# Plot the points of the Planets' data
plt.plot(planets_data[:,0],planets_data[:,1], 'x')

# Save a figure for the Plotting of the Planets' data points
# NOTE: This needs to be done always after the plot function
#       and before the close function
plt.savefig('../files/imgs/planets-data-2.png',dpi=300)

# Show the Plot, containing
# the points of the Planets' data
plt.show()

# Close the Figure for Plotting
plt.close()

# Disable the Matplotlib command, in line,
# for IPython's interactive console
# Note: Uncomment, if you want
#%matplotlib inline