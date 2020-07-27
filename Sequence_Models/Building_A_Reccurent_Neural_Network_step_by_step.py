import torch

# implement softmax and tanh in torch
softmax = lambda x : torch.exp(x)/torch.sum(torch.exp(x))
tanh = lambda x : (torch.exp(2*x)-1)/(torch.exp(2*x)+1)
sigmoid = lambda x: torch.exp(x)/(torch.exp(x)+ 1)


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

#LSTM NETWORK

def lstm_cell_forward(xt,a_prev,c_prev,parameters):

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]

    n_x,m = xt.shape
    n_y,n_a = Wy.shape

    #Concatanate a_prev and xt
    concat = torch.cat((a_prev,xt))

    #Compute gates and states

    ft = sigmoid(Wf @ concat + bf)
    it = sigmoid(Wi @ concat + bi)
    cct = tanh(Wc @ concat + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(Wo @ concat + bo)
    a_next = ot * tanh(c_next)

    #Compute prediction of the LSTM cell
    yt_pred = softmax(Wy @ a_next + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it , cct, ot , xt, parameters)

    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
c_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

#convert to torch tensors

xt_tmp = torch.tensor(xt_tmp)
a_prev_tmp = torch.tensor(a_prev_tmp)
c_prev_tmp = torch.tensor(a_prev_tmp)

for k,v in parameters_tmp.items():
    parameters_tmp[k] = torch.tensor(v)


a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))

#LSTM forward

def lstm_forward(x,a0,parameters):
    caches =[]

    Wy = parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape

    #initialize "a","c","y" with zeros

    a = torch.zeros(n_a,m,T_x)
    c = torch.zeros(n_a,m,T_x)
    y = torch.zeros(n_y,m,T_x)

    a_next = a0
    c_next = torch.zeros((n_a,m))

    for t in range(T_x):
        xt = x[:,:,t]

        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward(xt,a_next,c_next,parameters)
        # Update a,c,y
        a[:,:,t] = a_next
        c[:,:,t] = c_next
        y[:,:,t] = yt

        caches.append(cache)

    caches = (caches,x)
    return a, y, c, caches

np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi']= np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

#Convert arrays to torch tensors

x_tmp = torch.tensor(x_tmp)
a0_tmp = torch.tensor(a0_tmp)
for k,v in parameters_tmp.items():
    parameters_tmp[k] = torch.tensor(v)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))