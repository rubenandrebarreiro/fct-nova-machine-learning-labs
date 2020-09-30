# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:50:15 2020

@author: Ruben Andre Barreiro
"""

# Import matplotlib Python's Library
%matplotlib inline

# From Pylab Python's Library,
# Import all the Modules
from pylab import *

# Arange the Plot's scale, in the range [0.0 , 2.0],
# with an interval of 0.01
t = arange(0.0, 2.0, 0.01)

# Define the previously defined function as,
# s = sin( 2 * pi * t )
s = sin(2 * pi*t)

# Plot the previously defined Function
plot(t, s)

# Show the Plot of
# the previously defined Funtion
show()