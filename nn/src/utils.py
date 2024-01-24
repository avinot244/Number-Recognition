import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from globals import *

def count(elt, array):
    count = 0
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == elt : count+= 1
    return count

def reshapeFromShape(lst, shape):
    new_shape = 0
    for i in range(len(shape)):
        new_shape = new_shape+shape[i]
    return np.array(lst).reshape(new_shape)

def getShape(list):
    return [len(a) for a in list]

def graph(X,Y):
    plt.plot(X, Y)
    plt.ylabel("loss")
    plt.xlabel("iterations")
    plt.show()

def getPrediction(t):
    max = t[0]
    idx = 0
    for i in range(len(t)):
        if t[i] > max :
            max = t[i]
            idx = i
    return idx

