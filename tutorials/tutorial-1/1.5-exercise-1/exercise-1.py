# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:58:26 2020

@author: Ruben Andre Barreiro
"""

# Import NumPy Python's Library
import numpy as np

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

    return np.array(rows)

planets_data = load_planet_data('../files/planets.csv')



# G is the gravitational constant
# 6.67 * 10^(-11) N * m^2 * kg^(-2)
gravitational_constant = 6.67 * 1e-11

# One AU is 1.496 * 10^(11) meters
orbital_radius_au_in_meters = 1.496 * 1e11

# One Earth year is 3.16 * 10^7 seconds long
orbital_period_in_earth_years_seconds = 3.16 * 1e7



# Print the Orbital Radius/Semi-Major Axis (AU) and
# the Orbital Period (Julian/Earth Years) for all Planets
print('\n')
print('The Planets\' data, containing\nthe Orbital Radius/Semi-Major Axis (AU) and\nthe Orbital Periods (Julian/Earth Years):')
print(planets_data)



# Select the Orbital Radius/Semi-Major Axis (AU) of the Mercury Planet
mercury_orbital_radius_in_au = planets_data[0,0]

# Select the Orbital Period (Julian/Earth Years) of the Mercury Planet
mercury_orbital_period_in_earth_years = planets_data[0,1]



# Print the Orbital Radius/Semi-Major Axis (AU) of the Mercury Planet
print('\n')
print('Now, let\'s select the Orbital Radius/Semi-Major Axis (AU) of\nthe Mercury Planet:')
print(mercury_orbital_radius_in_au)

# Print the Orbital Period (Julian/Earth Years) of the Mercury Planet
print('\n')
print('And then, let\'s select the Orbital Period (Julian/Earth Years) of\nthe Mercury Planet:')
print(mercury_orbital_period_in_earth_years)



# The Orbital Radius/Semi-Major Axis (Meters) of the Mercury Planet
mercury_orbital_radius_in_meters = mercury_orbital_radius_in_au * orbital_radius_au_in_meters

# The Orbital Period (Seconds) of the Mercury Planet
mercury_orbital_period_in_seconds = mercury_orbital_period_in_earth_years * orbital_period_in_earth_years_seconds



# Print the Orbital Radius/Semi-Major Axis (Meters) of the Mercury Planet
print('\n')
print('The Orbital Radius/Semi-Major Axis (Meters) of\nthe Mercury Planet:')
print(mercury_orbital_radius_in_meters)

# Print the Orbital Period (Seconds) of the Mercury Planet
print('\n')
print('And then, let\'s select the Orbital Period (Seconds) of\nthe Mercury Planet:')
print(mercury_orbital_period_in_seconds)



# The Orbital Velocity (Meters/Seconds) of the Mercury Planet
mercury_planet_orbital_velocity_in_meters_by_seconds = ( 2 * np.pi * mercury_orbital_radius_in_meters ) / mercury_orbital_period_in_seconds



# Print the Orbital Velocity (Meters/Seconds) of the Mercury Planet
print('\n')
print('Computing the Orbital Velocity (Meters/Seconds) of\nthe Mercury Planet, we got:')
print(mercury_planet_orbital_velocity_in_meters_by_seconds)



# The Mass of the Sun, from the Mercury Planet
mass_sun_from_mercury_planet = ( mercury_planet_orbital_velocity_in_meters_by_seconds**2 * mercury_orbital_radius_in_meters ) / gravitational_constant



# Print the Mass of the Sun, from the Mercury Planet
print('\n')
print('Computing the Mass of the Sun, from the Mercury Planet, we got:')
print(mass_sun_from_mercury_planet)



# -------------------------------------------------------------



# Now, let's try a generalization for all Planets

# Select the Orbital Radius/Semi-Major Axis (AU) of all the Planets
planets_orbital_radius_in_au = planets_data[:,0]

# Select the Orbital Period (Julian/Earth Years) of all the Planets
planets_orbital_period_in_earth_years = planets_data[:,1]



# Print the Orbital Radius/Semi-Major Axis (AU) of all the Planets
print('\n')
print('Now, let\'s select the Orbital Radius/Semi-Major Axis (AU) of\nall the Planets:')
print(planets_orbital_radius_in_au)

# Print the Orbital Period (Julian/Earth Years) of all the Planets
print('\n')
print('And then, let\'s select the Orbital Period (Julian/Earth Years) of\nall the Planets:')
print(planets_orbital_period_in_earth_years)



# The Orbital Radius/Semi-Major Axis (Meters) of all the Planets
planets_orbital_radius_in_meters = planets_orbital_radius_in_au * orbital_radius_au_in_meters

# The Orbital Period (Seconds) of all the Planets
planets_orbital_period_in_seconds = planets_orbital_period_in_earth_years * orbital_period_in_earth_years_seconds



# Print the Orbital Radius/Semi-Major Axis (Meters) of all the Planets
print('\n')
print('The Orbital Radius/Semi-Major Axis (Meters) of\nall the Planets:')
print(planets_orbital_radius_in_meters)

# Print the Orbital Period (Seconds) of all the Planets
print('\n')
print('And then, let\'s select the Orbital Period (Seconds) of\nall the Planets:')
print(planets_orbital_period_in_seconds)



# The Orbital Velocity (Meters/Seconds) of all the Planets
planets_orbital_velocity_in_meters_by_seconds = ( 2 * np.pi * planets_orbital_radius_in_meters ) / planets_orbital_period_in_seconds



# Print the Orbital Velocity (Meters/Seconds) of all the Planets
print('\n')
print('Computing the Orbital Velocity (Meters/Seconds) of\nall the Planets, we got:')
print(planets_orbital_velocity_in_meters_by_seconds)



# The Mass of the Sun, from all the Planets
mass_sun_from_all_planets = ( planets_orbital_velocity_in_meters_by_seconds**2 * planets_orbital_radius_in_meters ) / gravitational_constant



# Print the Mass of the Sun, from all the Planets
print('\n')
print('Computing the Mass of the Sun, from all the Planets, we got:')
print(mass_sun_from_all_planets)



# The Mean/Average of the Mass of the Sun, from all the Planets
mass_sun_from_all_planets_mean_average = np.mean(mass_sun_from_all_planets)

# The Standard Deviation of the Mass of the Sun, from all the Planets
mass_sun_from_all_planets_standard_deviation = np.std(mass_sun_from_all_planets)



# Print the Mean/Average of the Mass of the Sun, from all the Planets
print('\n')
print('The Mean/Average of the Mass of the Sun, from all the Planets:')
print(mass_sun_from_all_planets_mean_average)

# Print the Standard Deviation of the Mass of the Sun, from all the Planets
print('\n')
print('The Standard Deviation of the Mass of the Sun, from all the Planets:')
print(mass_sun_from_all_planets_standard_deviation)


# Import Matplotlib Python's Library
import matplotlib.pyplot as plt

# Set the X and Y values, for the Plotting
x = planets_orbital_radius_in_au
y = planets_orbital_period_in_earth_years

# Set the Coefficients, of Degree 2
coefs = np.polyfit(x,y,2)

# Set the Pixels, for the Linear Space
pxs = np.linspace(0,max(x),100)

# Fit a 2nd Degree Polynomial function
poly = np.polyval(coefs,pxs)

# Create a Figure for Plotting
plt.figure(figsize=(12, 8))

# Plot the points of the Planets' data:
# 1) X - Orbital Radius (AUs)
# 2) Y - Orbital Period (Earth's Years)
plt.plot(x,y,'or')

# Plot the Fitted 2nd Degree Polynomial Function
plt.plot(pxs,poly,'-')

# Set the Axis for the Plotting Chart
plt.axis([0,max(x) + 5,-50,200])

# Set the Title for the Plotting Chart
plt.title('Degree: 2')

# Save a figure for the Plotting of the Planets' data points
# NOTE: This needs to be done always after the plot function
#       and before the close function
plt.savefig('../files/imgs/planets-data-3.png',dpi=300)

# Show the Plot, containing
# the points of the Planets' data and the Fitted 2nd Degree Polynomial Function
plt.show()

# Close the Figure for Plotting
plt.close()