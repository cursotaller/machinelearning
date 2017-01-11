#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    =========================================================
    Linear Regression Example
    =========================================================
    This example uses only the first feature of the `diabetes` dataset, in
    order to illustrate a two-dimensional plot of this regression technique. The
    straight line can be seen in the plot, showing how linear regression attempts
    to draw a straight line that will best minimize the residual sum of squares
    between the observed responses in the dataset, and the responses predicted by
    the linear approximation.
    
    The coefficients, the residual sum of squares and the variance score are also
    calculated.
    
    """
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

#print diabetes.data
#print diabetes.target

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis] #The newaxis object can be used to create an axis of length one.
diabetes_X_temp = diabetes_X[:, :, 2] #It takes the third column

print 'diabetes_X_temp'
print diabetes_X_temp

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:] #The last 20 elements of diabetes_X_temp

print 'diabetes_X_train'
print diabetes_X_train
print 'diabetes_X_test'
print diabetes_X_test

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

print 'diabetes_Y_train'
print diabetes_y_train
print 'diabetes_Y_test'
print diabetes_y_test

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and Y.
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()