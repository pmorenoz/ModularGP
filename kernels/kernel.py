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



import torch
import numpy as np
from util import squared_distance

class Kernel(torch.nn.Module):
    """
    Base class for kernels
    """
    def __init__(self, input_dim=None):
        super(Kernel, self).__init__()

        # Input dimension -- x
        if input_dim is None:
            input_dim = 1
        else:
            input_dim = int(input_dim)

        self.input_dim = input_dim