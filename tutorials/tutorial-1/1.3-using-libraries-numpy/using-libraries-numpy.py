# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:30:14 2020

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

# Exercise #1:
# - Select the rows corresponding to
#   Planets with Orbital Periods greater than 10 years
planets_with_orbital_periods_greater_than_10_years = planets_data[planets_data[:,1]>10,:]

print('Planets with Orbital Periods greater than 10 years:\n')
print(planets_with_orbital_periods_greater_than_10_years)
print('\n')

# Exercise #2:
# - How many planets have Orbital Periods greater than 10 years?
# - Hint: You can use the len function
#         to find the length of an array or the numpy.sum function to find the total sum of array elements,
#         which considers True to be 1 and False to be 0
how_many_planets_with_orbital_periods_greater_than_10_years = numpy.sum(planets_data[:,1]>10)

print('How many Planets with Orbital Periods greater than 10 years:\n')
print(how_many_planets_with_orbital_periods_greater_than_10_years)
print('\n')

# Exercise #3:
# - Select the Orbital Periods of the Planets,
#   whose Orbital Periods in years, are greater than
#   twice the Orbital Radius, in AU
# - Note that you can use algebraical operators,
#   such as sum or multiplication, with array objects
planets_orbital_periods_greater_than_twice_the_orbital_radius_in_au = planets_data[planets_data[:,1]>(2*planets_data[:,0]),1]

print('Orbital Periods of the Planets, whose Orbital Periods, in years, are greater than twice the Orbital Radius, in AU:\n')
print(planets_orbital_periods_greater_than_twice_the_orbital_radius_in_au)
print('\n')