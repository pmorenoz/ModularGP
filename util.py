# squared distance is based on the gptorch code
# by Steven Atkinson (steven@atkinson.mn)
# -----------------------------------------------------------------
# This script belongs to the ModularGP repo
# "Modular Gaussian Processes for Transfer Learning" @ NeurIPS 2021
# Copyright (c) 2021 Pablo Moreno-Munoz
# -----------------------------------------------------------------
#
#
# Pablo Moreno-Munoz (pabmo@dtu.dk)
# Section for Cognitive Systems
# Technical University of Denmark (DTU)
# October 2021

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

_lim_val = np.finfo(np.float64).max
_lim_val_exp = np.log(_lim_val)
_lim_val_square = np.sqrt(_lim_val)
#_lim_val_cube = cbrt(_lim_val)
_lim_val_cube = np.nextafter(_lim_val**(1/3.0), -np.inf)
_lim_val_quad = np.nextafter(_lim_val**(1/4.0), -np.inf)
_lim_val_three_times = np.nextafter(_lim_val/3.0, -np.inf)


def safe_exp(f):
    clamp_f = torch.clamp(f, min=-np.inf, max=_lim_val_exp)
    return torch.exp(clamp_f)

def safe_square(f):
    f = torch.clamp(f, min=-np.inf, max=_lim_val_square)
    return f**2

def safe_cube(f):
    f = torch.clamp(f, min=-np.inf, max=_lim_val_cube)
    return f**3

def safe_quad(f):
    f = torch.clamp(f, min=-np.inf, max=_lim_val_quad)
    return f**4

def true_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        5*torch.cos(7*np.pi*x + 2.4*np.pi)
    return y

def smooth_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi)
    return y

def smooth_function_bias(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        3.0*x - 7.5
    return y


def true_u_functions(x_list, Q):
    u_functions = []
    amplitude = (1.5 - 0.5) * torch.rand(Q, 3) + 0.5
    freq = (3 - 1) * torch.rand(Q, 3) + 1
    shift = 2 * torch.rand(Q, 3)
    for x in x_list:
        u_function = torch.empty(x.shape[0], Q)
        for q in range(Q):
            u_function[:,q,None] = 3.0 * amplitude[q, 0] * np.cos(freq[q, 0] * np.pi * x + shift[q, 0] * np.pi) - \
                                     2.0 * amplitude[q, 1] * np.sin(2 * freq[q, 1] * np.pi * x + shift[q, 1] * np.pi) + \
                                     amplitude[q, 2] * np.cos(4 * freq[q, 2] * np.pi * x + shift[q, 2] * np.pi)
        u_functions.append(u_function)
    return u_functions


def true_f_functions(x_list, Q):
    K = len(x_list)
    W = 0.5 * torch.randn(K, Q)
    f_functions = []
    u_functions = true_u_functions(x_list, Q)
    for k, u_function in enumerate(u_functions):
        Nk = u_function.shape[0]
        f_function = torch.zeros(Nk, 1)
        for q in range(Q):
            f_function += torch.tile(W[k:k+1, q:q+1], (Nk, 1)) * u_function[:, q:q+1]

        f_functions.append(f_function)

    return f_functions


def squared_distance(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)

    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)

    r2 = x1s + x2s.t() -2.0 * x1 @ x2.t()

    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()


class DataGP(Dataset):
    def __init__(self, x, y):
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class DataMOGP(Dataset):
    def __init__(self, x, y):
        self.x = x  # x is a list
        self.y = y  # y is a list

    def __len__(self):
        return min(len(x_d) for x_d in self.x)

    def __getitem__(self, item):
        x_tuple = tuple(x_d[item] for x_d in self.x)
        y_tuple = tuple(y_d[item] for y_d in self.y)
        return x_tuple, y_tuple



