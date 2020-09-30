# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:16:07 2020

@author: Ruben Andre Barreiro
"""

# Open the File 'planets.csv'
file = open('../files/planets.csv')

# Read the Lines, from the File 'planets.csv' 
lines = file.readlines()

# Close the File 'planets.csv',
# after read their lines
file.close()

# Read a file in a single line of code
# Note: Uncomment it, if you want
#lines = open('planets.csv').readlines()

# Echo the lines, from the File opened previously
lines

# Print the lines, from the File opened previously
print(lines)


# Return the Orbital Period from a Planet
# Note: Uses the 'planets.csv' file
def planet_period(planet):
      
    lines = open('../files/planets.csv').readlines()
    
    for line in lines:
        parts = line.split(',')
        
        if ( parts[0] == planet ):
            return float(parts[2])