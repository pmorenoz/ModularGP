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
from util import squared_distance
from kernels.kernel import Kernel
from kernels.rbf import RBF

class LMC(Kernel):
    """
    Class for Linear Model of Coregionalization / Kernel
    """

    def __init__(self, kernels, output_dim, rank=1, W=None, kappa=None, variance=None, length_scale=None, input_dim=None):
        super().__init__(input_dim)

        # Dimensionality of coregionalization kernel
        self.Q = len(kernels)
        self.output_dim = output_dim
        self.rank = rank
        if self.rank > output_dim:
            print("Warning: Unusual choice of rank, rank should be less than output dim.")

        # Coregionalization kernel / mixing hyper-parameters
        if W is None:
            self.W = torch.nn.Parameter(torch.randn(self.output_dim, self.Q), requires_grad=True)
        else:
            assert W.shape == (self.output_dim, self.Q, self.rank)

        # Registration of coregionalization parameters
        self.register_parameter('coregionalization_W', self.W)

        # Independent kernels
        self.kernels = kernels

    def B_coefficients(self):
        B_coeff = []
        for q in range(self.Q):
            B_q = torch.mm(self.W[:,q:q+1], self.W[:,q:q+1].t())
            B_coeff.append(B_q)
        return B_coeff

    def Kff(self, X, k):
        """
        Builds the cross-covariance matrix Kfdfd = cov[f_d(x),f_d(x)] of a Multi-output GP
        :param X: Input data
        :param k: Output function
        """
        N,_ = X.shape
        Kff = torch.zeros(N,N)
        B = self.B_coefficients()
        for q, B_q in enumerate(B):
            Kff += B_q[k,k] * self.kernels[q].K(X, X)

        return Kff

    def Kfu(self, X, Z, k):
        """
        Builds the cross-covariance cov[f_d(x),u(z)] of a Multi-output GP
        :param X: Input data
        :param Z: Inducing points (M, D, Q)
        :param k: Output function
        """
        N, _ = X.shape
        M, Xdim, _ = Z.shape

        B = self.B_coefficients()
        Kfu = torch.empty(N, M, self.Q)
        for q, B_q in enumerate(B):
            Kfu[:,:,q] = self.W[k,q] * self.kernels[q].K(X, Z[:,:,q])

        return Kfu

