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

class Likelihood(torch.nn.Module):
    """
    Base class for likelihoods
    """
    def __init__(self):
        super(Likelihood, self).__init__()

    def gh_points(self, T=20):
        # Gaussian-Hermite Quadrature points
        gh_p, gh_w = np.polynomial.hermite.hermgauss(T)
        gh_p, gh_w = torch.from_numpy(gh_p), torch.from_numpy(gh_w)
        return gh_p, gh_w
