import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#standardize the dataset

train_set_x = train_set_x_flatten/255.
test_set_x  = test_set_x_flatten/255.

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

#Initialize weights

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))

    b = np.int(0)

    assert (w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b

#Forward and Backward propagation

def propagate(w,b,X,Y):
    m = X.shape[1]
    Z = w.T @ X + b
    A = sigmoid(Z)
    cost = (-1/m)*np.sum(Y * np.log(A) + (1-Y)*(np.log(1-A)))

    dw = (1/m)*X@(A-Y).T
    db = (1/m)*(A-Y).sum()

    grads = {"dw":dw,"db":db}

    return grads,cost

def optimize(w,b,X,Y,num_iterations, learning_rate,print_cost = False):

    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)

        dw,db = grads.values()
        w = w-learning_rate*dw
        b = b-learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        if print_cost and i % 100 ==0:
            print("Cost after iteration %i: %f" %(i,cost))
    params = {"w":w,"b":b}
    grads  = {"dw":dw,"db":db}

    return params,grads,costs

def predict(w,b,X):
    m = X.shape[1]
    A = sigmoid(w.T @ X + b)
    A[A>0.5]=1
    A[A <= 0.5]=0
    return A


#Merge all functions into a model

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,
          print_cost= False):

    w,b = initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=print_cost)

    w,b = parameters.values()

    Y_pred_train = predict(w,b,X_train)
    Y_pred_test  = predict(w,b,X_test)

    print("train accuracy: {}".format(100- 100*np.mean(np.abs(Y_pred_train-Y_train))))
    print("test accuracy: {}".format(100- 100*np.mean(np.abs(Y_pred_test-Y_test))))

    history = {"costs":costs,
               "Y_pred_test":Y_pred_test,
               "Y_pred_train":Y_pred_train,
               "w":w,"b":b,
               "learning_rate":learning_rate,
               "num_iter":num_iterations}
    return history

history = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
