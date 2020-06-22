import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



#fucntion that returns the array padded
def zero_pad(X,pad):
    X_pad = F.pad(X,(0,0,pad,pad,pad,pad,0,0),'constant',0)

    return X_pad

x = torch.tensor(np.random.randn(4,3,3,2))
x_pad = zero_pad(x,2)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

fig,axarr = plt.subplots(1,2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x pad')
axarr[1].imshow(x_pad[0,:,:,0])

# convolution operation result over a single slice
def conv_single_step(a_slice_prev,W,b):
    s = a_slice_prev * W
    z = torch.sum(s)
    z = float(z+b)
    return z


np.random.seed(1)

a_slice_prev =torch.tensor( np.random.randn(4, 4, 3))
W = torch.tensor(np.random.randn(4, 4, 3))
b =torch.tensor( np.random.randn(1, 1, 1))

Z = conv_single_step(a_slice_prev,W,b)
print(Z)

# returns resulting tensor after convolution and copy of tensors required for the the backpropagation
def conv_forward(A_prev,W,b,hparams):
    m,n_H_prev,n_W_prev,n_c_prev = A_prev.shape
    f,f,n_c_prev,n_c = W.shape
    stride,pad = hparams['stride'],hparams['pad']
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = torch.zeros(m,n_H,n_W,n_c)
    A_prev_pad = zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            vertical_start = h*stride
            vertical_end = vertical_start + f
            for w in range(n_W):
                horiz_start = w*stride
                horiz_end = horiz_start + f

                for c in range(n_c):
                    a_slice_prev = a_prev_pad[vertical_start:vertical_end,horiz_start:horiz_end,:]
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z [i,h,w,c] = conv_single_step (a_slice_prev,weights,biases)

    assert (Z.shape == (m,n_H,n_W,n_c))

    cache = (A_prev,W,b,hparams)
    return Z,cache
