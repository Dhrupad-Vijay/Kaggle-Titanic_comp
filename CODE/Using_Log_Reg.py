########################################################################################################################
#                                                   Titanic - Kaggle challenge [Logistic Regression]
########################################################################################################################
# used for manipulating directory paths
import os
import pandas as pd
# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import random

########################################################################################################################

path_train = open('/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/train1.csv')          #Training data path
array = np.loadtxt(path_train, delimiter=',',skiprows=1)

path_test = open('/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/test1.csv')            #Test data path
array1 = np.loadtxt(path_test, delimiter=',',skiprows=1)

no_fetures = len(array[0])-1
x = array[:,0:no_fetures]
x_test = array1[:,0:no_fetures]

y = np.transpose([array[:,no_fetures]])
y_test = np.transpose([array1[:,no_fetures]])
m,n = x.shape
mt,nt = x_test.shape
#print(y.shape)
theta = np.zeros((n+1,1)) # theta is a 891 X 1 matrix
###################################################################################################################
# Normalization function

def normalize(x):
    mu = x.mean(axis = 0)
    temp1 = x-mu  
    sigma = np.std(temp1,axis = 0)
    temp2 = temp1/sigma
    X = np.c_[np.ones((m,1)),temp2]       # X is a 891 X 7 matrix
    na = [X,mu,sigma]
    return na

###################################################################################################################
# sigmoid function

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

###################################################################################################################

# Random weight initialization

def randInitializeWeights(theta,epsilon_init=0.12):
  
    W = np.zeros((len(theta),1))
    epsilon_init = 0.12
    W = np.random.rand(len(theta),1) * 2 * epsilon_init - epsilon_init

    return W

###################################################################################################################
# Logistic regression

def log_reg(theta, X, y, lambda_):
    
    q = 1
    theta = theta[:,np.newaxis]     #<---- THIS IS THE TRICK
    grad = np.zeros(theta.shape)
    n = len(theta)

    z = X.dot(theta)
    J = (1/m) * ((-y.T).dot(np.log(sigmoid(z))) - ((1-y).T).dot(np.log(1-sigmoid(z)))) + (lambda_/(2*m))*(theta[2:n].T.dot(theta[2:n]))

    grad = (1/m) * ((X.T).dot(sigmoid(z) - y))
    grad[q:n+1] = grad[q:n+1] + (lambda_/m) * theta[q:n+1]

    return J, grad

###################################################################################################################

#temp1 = normalize(x)
X = np.c_[np.ones((m,1)),x]
#X = temp1[0]
#lambda_ = np.array([1,2,3,4,5])
lambda_ = 2

#initial_theta = np.zeros((n+1))

initial_theta = randInitializeWeights(theta,epsilon_init=0.12)

# Set number of iterations

options= {'maxiter': 5000}
initial_theta=np.zeros(n+1)

J, grad = log_reg(initial_theta,X,y,lambda_)

# scipy optimization

import scipy.optimize as opt
model = opt.minimize(fun = log_reg, x0 = initial_theta, args = (X, y, lambda_), method = 'TNC', jac =True, options=options)


# get the solution of the optimization final result
theta = model.x
a = np.zeros((m,1))
survival = X.dot(theta)

for i in range(len(survival)):
  if(survival[i]>0):
    a[i] = 1
  else:
    a[i] = 0
print("Train accuracy is = ",np.mean(a==y)*100,'%')

# accuracy of the test

Xt = np.c_[np.ones((mt,1)),x_test]
surv = Xt.dot(theta)
b = np.zeros((mt,1))

for i in range(len(surv)):
  if(surv[i]>0):
    b[i] = 1
  else:
    b[i] = 0

import pandas as pd
df = pd.DataFrame(b)
df.to_excel(excel_writer = "/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/my_submission.xlsx")       #Save the result in a excel file
