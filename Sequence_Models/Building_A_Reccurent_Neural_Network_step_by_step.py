import torch

# implement softmax and tanh in torch
softmax = lambda x : torch.exp(x)/torch.sum(torch.exp(x))
tanh = lambda x : (torch.exp(2*x)-1)/(torch.exp(2*x)+1)



def rnn_cell_forward(xt, a_prev, parameters):

    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    ### START CODE HERE ### (≈2 lines)
    # compute next activation state using the formula given above
    a_next = tanh(Waa @ a_prev + Wax @ xt + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(Wya @ a_next + by)
    ### END CODE HERE ###

    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

def rnn_forward(x,a0,parameters):


    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    ### START CODE HERE ###

    # initialize "a" and "y_pred" with zeros (≈2 lines)
    a = torch.zeros((n_a, m, T_x))
    y_pred = torch.zeros((n_y, m, T_x))

    # Initialize a_next (≈1 line)
    a_next = a0

    # loop over all time-steps of the input 'x' (1 line)
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈2 lines)
        xt = x[:, :, t]
        a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)

    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


x_tmp = torch.rand(3,10,4)
a0_tmp = torch.rand(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = torch.rand(5,5)
parameters_tmp['Wax'] = torch.rand(5,3)
parameters_tmp['Wya'] = torch.rand(2,5)
parameters_tmp['ba'] = torch.rand(5,1)
parameters_tmp['by'] = torch.rand(2,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))