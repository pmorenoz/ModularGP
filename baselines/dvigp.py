# Implementation of the "Distributed Variational Inference in GPs"
# by Y. Gal and M. van der Wilk
#
# Little adaptation without the LVM assumption
# for testing and comparison. Simulates a distributed environment.
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
from torch.distributions import MultivariateNormal as Normal
from torch.distributions import kl_divergence

from GPy.inference.latent_function_inference.posterior import Posterior

class DVIGP(torch.nn.Module):
    """
    -- Distributed Variational Inference in Gaussian Processes --
    --
    -- Adaptation to Pytorch + GP framework
    -- Y. Gal et al. "Distributed Variational Inference in Sparse Gaussian
                      Process Regression and Latent Variable Models" NIPS 2014
    """
    def __init__(self, kernel, likelihood, M, nodes=1, input_dim=None):
        super(DVIGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        # Nodes to distribute the computational load --
        self.nodes = int(nodes)

        # Dimensions --
        self.M = M                          #num. inducing
        self.input_dim = int(input_dim)     #dimension of x

        # GP Elements --
        self.likelihood = likelihood        #type of likelihood
        self.kernel = kernel                #type of kernel
        self.z = torch.nn.Parameter(torch.linspace(-0.9, 0.9, self.M)[:,None], requires_grad=False)

        # Variational distribution --
        self.q_m = torch.nn.Parameter(torch.randn(M,1), requires_grad=True)  # variational: mean parameter
        self.q_L = torch.nn.Parameter(torch.eye(M), requires_grad=True)  # variational: covariance

    def forward(self, x, y):
        x_nodes, y_nodes = self.data_to_nodes(x,y)

        # Variational parameters --
        q_m = self.q_m
        q_L = torch.tril(self.q_L)
        q_S = torch.mm(q_L, q_L.t())

        # Prior parameters (uses kernel) --
        Kuu = self.kernel.K(self.z)
        iKuu, _ = torch.solve(torch.eye(self.M), Kuu)  # is pseudo-inverse?

        # Distributions -- q(u), p(u)
        q_u = Normal(q_m.flatten(), q_S)
        p_u = Normal(torch.zeros(self.M), Kuu)

        global_params = {'q_m': q_m, 'q_L': q_L, 'q_S': q_S, 'Kuu': Kuu, 'iKuu': iKuu}

        # Distributed Expectations
        expectation = 0.0
        for k, y_k in enumerate(y_nodes):
            x_k = x_nodes[k]
            expectation_node = self.forward_node(x_k, y_k, global_params)
            expectation += expectation_node.sum()

        # KL divergence --
        kl = kl_divergence(q_u, p_u)

        # Lower bound (ELBO) --
        elbo = expectation - kl

        return -elbo

    def forward_node(self, x_node, y_node, global_params):
        q_m = global_params['q_m']
        q_L = global_params['q_m']
        q_S = global_params['q_S']
        Kuu = global_params['Kuu']
        iKuu = global_params['iKuu']

        Kff = self.kernel.K(x_node, x_node)
        Kfu = self.kernel.K(x_node, self.z)
        Kuf = torch.transpose(Kfu, 0, 1)

        A = Kfu.mm(iKuu)
        AT = iKuu.mm(Kuf)

        m_f = A.mm(q_m)
        v_f = torch.diag(Kff + A.mm(q_S - Kuu).mm(AT))

        # Expectation term of node --
        expectation = self.likelihood.variational_expectation(y_node, m_f, v_f)

        return expectation

    def data_to_nodes(self, x, y):
        x_nodes = []
        y_nodes = []

        N = y.size(0)
        size_node = np.int(np.floor(N/self.nodes))
        for k in range(self.nodes):
            if k < self.nodes - 1:
                x_nodes.append(x[(0+(k*size_node)):(0+((k+1)*size_node)), :])
                y_nodes.append(y[(0+(k*size_node)):(0+((k+1)*size_node)), :])
            else:
                x_nodes.append(x[(0+(k*size_node)):, :])
                y_nodes.append(y[(0+(k*size_node)):, :])

        return x_nodes, y_nodes

    def predictive(self, x_new):
        # Matrices
        q_m = self.q_m.detach().numpy()
        q_L = torch.tril(self.q_L)
        q_S = torch.mm(q_L, q_L.t()).detach().numpy()
        Kuu = self.kernel.K(self.z, self.z).detach().numpy()

        posterior = Posterior(mean=q_m, cov=q_S, K=Kuu, prior_mean=np.zeros(q_m.shape))
        Kx = self.kernel.K(self.z, x_new).detach().numpy()
        Kxx = self.kernel.K(x_new, x_new).detach().numpy()

        # GP Predictive Posterior - mean + variance
        gp_mu = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        gp_var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None, :, :], 1)).T

        gp = gp_mu
        gp_upper = gp_mu + 2*np.sqrt(gp_var) #+ 2 * self.likelihood.sigma.detach().numpy()
        gp_lower = gp_mu - 2*np.sqrt(gp_var) #- 2 * self.likelihood.sigma.detach().numpy()

        return gp, gp_upper, gp_lower

    def rmse(self, x_new, f_new):
        f_gp,_,_ = self.predictive(x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x_new, f_new):
        f_gp,_,_ = self.predictive(x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, x_new, y_new):
        f_gp, u_gp, _ = self.predictive(x_new)
        f_gp = torch.from_numpy(f_gp)
        u_gp = torch.from_numpy(u_gp)
        v_gp = torch.pow(0.5*(u_gp - f_gp), 2.0)
        nlpd = - torch.mean(self.likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd

