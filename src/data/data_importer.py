import torch
import numpy as np
import matplotlib.pyplot as plt

# from src import device
from data.binvox_rw import *


# function to load 2D data
def load_data_2D(object_name):
    return torch.tensor(plt.imread(object_name)[:, :, 3] >= 0.5, device='cuda:0').float()


def load_binvox(fn):
    with open(fn, 'rb') as fin:
        out = read_as_3d_array(fin)
        return np.array(out.data)


# function to load 3D data
def load_data_3D(object_name):
    numpy_array = np.where(load_binvox(object_name) == True, 1.0, 0.0)
    return torch.tensor(numpy_array, dtype=torch.float32, device='cuda:0')


# function to load 4D data
def load_data_4D(object_name):
    numpy_array = np.load(object_name)
    return torch.tensor(numpy_array, dtype=torch.float32, device='cuda:0')

