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
from kernels.stationary import Stationary

class RBF(Stationary):
    """
    The Radial Basis Function (RBF) or Squared Exponential / Gaussian Kernel
    """

    def K(self, X, X2=None):
        variance = self.variance.abs().clamp(min=0.0, max=5.0)
        r2 = torch.clamp(self.squared_dist(X, X2),min=0.0, max=np.inf)
        K = variance*torch.exp(-r2 / 2.0)

        # Assure that is PSD
        if X2 is None:
            try:
                _ = torch.cholesky(K)
            except RuntimeError:
                print('Jitter added')
                jitter = 1e-5
                idx = torch.arange(K.shape[-1])
                Kprime = K.clone()
                Kprime[idx, idx] += jitter
                K = Kprime

        return K