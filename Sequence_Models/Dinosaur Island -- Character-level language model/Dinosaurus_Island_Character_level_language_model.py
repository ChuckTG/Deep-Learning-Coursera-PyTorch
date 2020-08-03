import numpy as np
from utils import *
import random
import pprint
import torch


data = open('dinos.txt','r').read()
data = data.lower()
chars = list(set(data))
chars = sorted(chars)

char_to_ix = {c:i for i,c in enumerate(chars)}
ix_to_char = {i:c for i,c in enumerate(chars)}
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)

# Clip function

def clip(gradients,maxValue):
    '''
    Clips Gradients values between a minimum and a maximum

    :param gradients: dictionary with gradients
    :param maxValue: the value that sets the minimum and maximum (min = -max)
    :return: a dictionary with clipped gradients
    '''
    gradients_arr = list(gradients.values())
    gradients_names = list(gradients.keys())
    for gradient in gradients_arr:
        gradient[gradient>maxValue] = maxValue
        gradient[gradient<-maxValue] = -maxValue

    gradients = {k:v for k,v in zip(gradients_names,gradients_arr)}

    return gradients

# Test with a maxvalue of 10
mValue = 5
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db   = np.random.randn(5,1)*10
dby  = np.random.randn(2,1)*10
gradients = {"dWax":dWax,"dWaa":dWaa,"dWya":dWya,"db":db,"dby":dby}
gradients = clip(gradients,mValue)
gradients = {k:torch.from_numpy(v) for k,v in gradients.items()}
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])