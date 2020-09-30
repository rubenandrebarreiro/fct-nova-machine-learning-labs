# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:20:19 2020

@author: Ruben Andre Barreiro
"""

# Import NumPy Python's Library
import numpy as np

# Import Matplotlib Python's Library
import matplotlib.pyplot as plt


# Load the Matrix with the Polynomial Data
mat = np.loadtxt('../files/polydata.csv',delimiter=';')


# Set the X values, from the Polynomial Data
x = mat[:,0]

# Set the Y values, from the Polynomial Data
y = mat[:,1]


# Set the Coefficients, of Degree 3
coefs_3 = np.polyfit(x,y,3)

# Set the Coefficients, of Degree 15
coefs_15 = np.polyfit(x,y,15)


# Set the Pixels, for the Linear Space
pxs = np.linspace(0,max(x),100)


# Fit a 3rd Degree Polynomial function
poly_3 = np.polyval(coefs_3,pxs)

# Fit a 15th Degree Polynomial function
poly_15 = np.polyval(coefs_15,pxs)


# Create a Figure for Plotting
plt.figure(figsize=(12, 8))

# Plot the points of the Polynomial data
plt.plot(x,y,'or')


# Plot the Fitted 3rd Degree Polynomial Function
plt.plot(pxs,poly_3,'-')

# Plot the Fitted 15th Degree Polynomial Function
plt.plot(pxs,poly_15,'-')


# Set the Axis for the Plotting Chart
plt.axis([0,max(x),-1.5,1.5])


# Set the Title for the Plotting Chart
#plt.title('Degree: 3')
#plt.title('Degree: 15')
plt.title('Degree: 3 and 15')

# Save a figure for the Plotting of the Polynomial data points
# NOTE: This needs to be done always after the plot function
#       and before the close function
#plt.savefig('../files/imgs/1.6-exercise-2-coef-3.png')
#plt.savefig('../files/imgs/1.6-exercise-2-coef-15.png')
plt.savefig('../files/imgs/1.6-exercise-2-coefs-3-and-15.png')


# Close the Figure for Plotting
plt.close()



# Print the Coefficients of the 3rd Degree Polynomial Function
print('\n')
print('The Coefficients of the 3rd Degree Polynomial Function:')
print(coefs_3)


# Print the Coefficients of the 15th Degree Polynomial Function
print('\n')
print('The Coefficients of the 15th Degree Polynomial Function:')
print(coefs_15)



# Exercise 1.6.1: 
# Q: What is the hypothesis class you are using in each case
#    (degree 3 and degree 15)?
# A: Degree 3: A polynomial curve of degree 3
#    Degree 15: A polynomial curve of degree 15
    

# Exercise 1.6.2: 
# Q: What is the corresponding model for each hypothesis class?
# A: Degree 3: y = θ_{1}x^{3} + θ_{2}x^{2} + θ_{3}x + θ_{4}
#    Degree 15: y = θ_{1}x^{15} + θ_{2}x^{14} + θ_{3}x^{13} + θ_{4}x^{12} +
#                 + θ_{5}x^{11} + θ_{6}x^{10} + θ_{7}x^{9} + θ_{8}x^{8} + 
#                 + θ_{9}x^{7} + θ_{10}x^{6} + θ_{11}x^{5} + θ_{12}x^{4} + 
#                 + θ_{13}x^{3} + θ_{14}x^{2} + θ_{15}x + θ_{16}
    

# Exercise 1.6.3: 
# Q: What is the hypothesis in each case?
# A: Degree 3:  [ θ_{1} = 0.12831139 ; θ_{2} = -1.24583487 ;
#                 θ_{3} = 3.01638739 ; θ_{4} = -1.10432872 ]
#    Degree 15: [ θ_{1} = -1.16565156e-03 ; θ_{2} = 5.94797587e-02 ;
#                 θ_{3} = -1.39485584e+00 ; θ_{4} = 1.99433035e+01 ;
#                 θ_{5} = -1.94455094e+02 ; θ_{6} = 1.36991372e+03 ;
#                 θ_{7} = -7.20494944e+03 ; θ_{8} = 2.88096907e+04 ;
#                 θ_{9} = -8.82893589e+04 ; θ_{10} = 2.07259302e+05 ;
#                 θ_{11} = -3.69289141e+05 ; θ_{12} = 4.89713241e+05 ;
#                 θ_{13} = -4.66923307e+05 ; θ_{14} = 3.01463118e+05 ;
#                 θ_{15} = -1.17541384e+05 ; θ_{16} = 2.08092837e+04 ]


# Exercise 1.6.4: 
# Q: How was each hypothesis chosen?
# A: Instantiating the parameters' vector (theta), representing the vector of
#    all theta_{1}; ... ; theta_{n} parameters.
#    The probability of the data given the hypothesis is
#    the likelihood of the hypothesis.
#    The best hypothesis is the one with the best (or maximum) likelihood
#    However, in some cases, increasing too much the degree of
#    a Polynomial Curve, can lead to not a good prediction of future values,
#    because the range of Polynomial Curve is very stricted to the data used
#    So, a good idea is to separate the global set in two subsets:
#    1) Training Set: The subset for training the Model;
#    2) Test Set: The subset for test the Model and predict the future values;