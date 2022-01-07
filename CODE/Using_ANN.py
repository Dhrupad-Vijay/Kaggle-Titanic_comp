########################################################################################################################
#                                                   Titanic - Kaggle challenge [ ANN ]
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

###################################################################################################################

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

###################################################################################################################

def sigmoidGradient(z):
    
    g = np.zeros(z.shape)

    g = sigmoid(z) * (1 - sigmoid(z))

    return g

###################################################################################################################

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
  
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W

###################################################################################################################

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
  
    # Reshape nn_params back into the parameters Theta1 (25*8) and Theta2 (2*26), the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                          (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                          (num_labels, (hidden_layer_size + 1)))
    
    #     Initialization and list of variables to calculate
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    y_mat = (np.array([0,1]) == y).T                            # generating y matrix
    X = np.c_[np.ones((m,1)),X]                                 # adding the bias

    #
    #====================== FORWARD PROPAGATION ========================
    #

    a1 = X.T
    z2 = Theta1.dot(a1)
    a2 = np.r_[ np.ones((1,m)) , sigmoid(z2)]
    z3 = Theta2.dot(a2)
    hx = sigmoid(z3)

    #
    #====================== COST FUNCTION ========================
    #

    reg = (lambda_/(2*m)) * (np.sum(pow(Theta1[:,1:] , 2)) + np.sum(pow(Theta2[:,1:] , 2)))
    J = (-1/m) * np.sum(y_mat * np.log(hx) + (1-y_mat) * np.log(1-hx)) + reg 

    #
    #====================== BACKWARD PROPAGATION ========================
    #

    delta3 = hx - y_mat
    z2 = np.r_[np.ones((1,m)),z2]
    delta2 = (Theta2.T.dot(delta3)) * (sigmoid(z2))
    delta2 = delta2[1:hidden_layer_size+1,:]

    Theta1_g = delta2.dot(a1.T)
    Theta2_g = delta3.dot(a2.T)

    Theta1_grad = (1 / m) * Theta1_g
    Theta2_grad = (1 / m) * Theta2_g

    Theta1_grad[:,1:] = (1) * ( Theta1_grad[:,1:] + (lambda_/m) * Theta1[:,1:] )
    Theta2_grad[:,1:] = (1) * ( Theta2_grad[:,1:] + (lambda_/m) * Theta2[:,1:] )

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

###################################################################################################################

def number_pred(x,Theta1,Theta2):
    
    mt = len(x)
    X = np.c_[np.ones((mt,1)),x]
    a1 = X.T
    z2 = Theta1.dot(a1)
    a2 = np.r_[ np.ones((1,mt)) , sigmoid(z2)]
    z3 = Theta2.dot(a2)
    hx = sigmoid(z3)
    p = np.argmax(hx, axis=1)
    
    return p

###################################################################################################################

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                   IMPORT TRAINING DATA [X y]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

path_train = open('/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/train1.csv')        # Training data path
array = np.loadtxt(path_train, delimiter=',',skiprows=1)

path_test = open('/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/test1.csv')          # Test data path
array1 = np.loadtxt(path_test, delimiter=',',skiprows=1)

no_fetures = len(array[0])-1
X_train = array[:,:no_fetures]
X_test = array1[:,:no_fetures]

y_train = np.transpose([array[:,no_fetures]])
y_test = np.transpose([array1[:,no_fetures]])
m,n = X_train.shape
mt,nt = X_test.shape

###################################################################################################################

# Setup the parameters you will use for this exercise
input_layer_size  = 7       # 7 Input Features
hidden_layer_size = 25      # 25 hidden units
num_labels = 2              # 2 labels, 0 & 1

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                             GENERATE NEURAL NETWORK PARAMETERS USING SCIPY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set number of iterations
options= {'maxiter': 500}

#  Regularization parameter lambda
lambda_ = 1.2
#t = np.array(random.sample(range(0, m), m))
#tt_split = 700
#X_train = X[t[:tt_split,],]
#X_test = X[t[tt_split:,],]
#y_train = y[t[:tt_split,],]
#y_test = y[t[tt_split:,],]

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X_train, y_train, lambda_)

# nncostFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

p = predict(Theta1, Theta2, X_test)

import pandas as pd
df = pd.DataFrame(p)
df.to_excel(excel_writer = "/content/drive/MyDrive/ML_&_AI/Kaggle/Titanic_Data/my_submission_ann.xlsx")       # Save the output in a excel file
