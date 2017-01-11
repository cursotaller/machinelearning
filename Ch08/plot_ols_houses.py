#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    =========================================================
    Linear Regression Example
    =========================================================
    This example uses only the first feature of the `mydata` dataset, in
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
from sklearn import linear_model

# Load the mydata dataset
mydata = np.loadtxt("houses.csv", delimiter=",")

mydata_X_temp = mydata[0:4]

print 'mydata_X_temp'
print mydata_X_temp

# Split the data into training/testing sets
mydata_X_train = mydata_X_temp[:-3]
mydata_X_test = mydata_X_temp[-3:] #The last 20 elements of mydata_X_temp

print 'mydata_X_train'
print mydata_X_train
print 'mydata_X_test'
print mydata_X_test

# Split the targets into training/testing sets
mydata_y_train = mydata.target[:-3]
mydata_y_test = mydata.target[-3:]

print 'mydata_Y_train'
print mydata_y_train
print 'mydata_Y_test'
print mydata_y_test

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(mydata_X_train, mydata_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(mydata_X_test) - mydata_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and Y.
print('Variance score: %.2f' % regr.score(mydata_X_test, mydata_y_test))

# Plot outputs
plt.scatter(mydata_X_test, mydata_y_test,  color='black')
plt.plot(mydata_X_test, regr.predict(mydata_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()