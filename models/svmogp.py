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
from torch.distributions import MultivariateNormal as Normal
from torch.distributions import kl_divergence
from kernels.coregionalization import LMC

import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior


class SVMOGP(torch.nn.Module):
    """
    -- Sparse Variational Multi-output Gaussian Process --
    --
    -- Adaptation to Pytorch + GP framework --
    -- Based on M. A. Álvarez and N. Lawrence, "Sparse convolved Gaussian processes for multi-output regression" NIPS'08
    -- Reference: http://papers.neurips.cc/paper/3553-sparse-convolved-gaussian-processes-for-multi-output-regression.pdf
    """
    def __init__(self, kernels, likelihoods, Q, M, input_dim=None, batch_rates=None):
        super(SVMOGP, self).__init__()

        if input_dim is None:
            input_dim = 1


        # Dimensions --
        self.M = M  # num. inducing
        self.Q = Q  # num. latent functions
        self.input_dim = int(input_dim)  # dimension of x

        # Likelihoods --
        self.likelihoods = likelihoods  # list of likelihoods
        self.D = len(self.likelihoods)  # num. output channels

        if batch_rates is None:
            self.batch_rates = self.D*[1.0]
        else:
            self.batch_rates = batch_rates

        # Kernels --
        self.kernels = torch.nn.ModuleList()
        for q in range(self.Q):
            self.kernels.append(kernels[q])
        self.coregionalization = LMC(self.kernels, self.D)  # is a list

        # Inducing points --
        if self.input_dim > 1:
            self.z = torch.nn.Parameter(torch.rand(self.M, self.input_dim, self.Q), requires_grad=False)
        else:
            self.z = torch.nn.Parameter(torch.tile(torch.linspace(0.1, 0.9, self.M)[:,None, None], (1, 1, self.Q)), requires_grad=False)


        # Variational distributions --
        self.q_m = torch.nn.Parameter(2*torch.randn(M, Q), requires_grad=True)  # variational: mean parameter
        self.q_L = torch.nn.Parameter(torch.tile(torch.eye(M)[:, :, None], (1, 1, self.Q)), requires_grad=True)  # variational: covariance

    def expectation(self, x, y):
        # Check length of input+output lists
        assert len(x) == self.D
        assert len(y) == self.D

        # MOGP prior + Variational parameters
        q_m = self.q_m
        q_S = torch.zeros(self.M, self.M, self.Q)
        Kuu = torch.zeros(self.M, self.M, self.Q)
        iKuu = torch.zeros(self.M, self.M, self.Q)

        for q in range(self.Q):
            # MOGP latent functions prior
            Kuu_q = self.kernels[q].K(self.z[:, :, q], self.z[:, :, q])
            iKuu_q, _ = torch.solve(torch.eye(self.M), Kuu_q)  # is pseudo-inverse?
            Kuu[:, :, q] = Kuu_q
            iKuu[:, :, q] = iKuu_q

            # Variational parameters + Gaussian integration
            q_L = torch.tril(self.q_L[:, :, q])
            q_S[:, :, q] = torch.mm(q_L, q_L.t())

        # Expectation values (NxD)
        expectation = []
        for d in range(self.D):
            Kff = self.coregionalization.Kff(x[d], d)
            Kfu = self.coregionalization.Kfu(x[d], self.z, d)

            m_f = 0.0
            S_f = Kff

            for q in range(self.Q):
                A = Kfu[:, :, q].mm(iKuu[:, :, q])
                AT = iKuu[:, :, q].mm(Kfu[:, :, q].t())

                m_f += A.mm(q_m[:, q:q + 1])
                S_f += A.mm(q_S[:, :, q]).mm(AT) - A.mm(Kfu[:, :, q].t())

            v_f = torch.diag(S_f)
            expectation.append(self.likelihoods[d].variational_expectation(y[d], m_f, v_f))

        return expectation

    def divergence(self, p_u, q_u):
        kl = 0.0
        for q in range(self.Q):
            kl += kl_divergence(q_u[q], p_u[q])
        return kl

    def forward(self, x, y):

        # Empty variables for filling in 1:Q
        q_u = []
        p_u = []
        q_m = self.q_m
        q_S = torch.zeros(self.M, self.M, self.Q)
        Kuu = torch.zeros(self.M, self.M, self.Q)
        for q in range(self.Q):

            # Variational parameters --
            q_L = torch.tril(self.q_L[:,:,q])
            q_S[:,:,q] = torch.mm(q_L, q_L.t())

            # Prior parameters (uses kernel) --
            Kuu_q = self.kernels[q].K(self.z[:, :, q], self.z[:, :, q])
            Kuu[:, :, q] = Kuu_q

            # Distributions -- q(u), p(u)
            q_u.append(Normal(q_m[:,q].flatten(), q_S[:,:,q]))
            p_u.append(Normal(torch.zeros(self.M), Kuu[:,:,q]))

        # Expectation term --
        expectation = 0.0
        expectation_mo = self.expectation(x, y)
        for d, exp in enumerate(expectation_mo):
            expectation += self.batch_rates[d] * exp.sum()

        # KL divergence --
        kl = self.divergence(q_u, p_u)

        # Lower bound (ELBO) --
        elbo = expectation - kl
        return -elbo

    def predictive(self, xnew, d):
        # MOGP prior + Variational parameters
        q_m = self.q_m
        q_S = torch.zeros(self.M, self.M, self.Q)
        Kuu = torch.zeros(self.M, self.M, self.Q)
        iKuu = torch.zeros(self.M, self.M, self.Q)

        # Posterior distribution on new input data
        Kff = self.coregionalization.Kff(xnew, d)
        Kfu = self.coregionalization.Kfu(xnew, self.z, d)

        m_pred = 0.0
        S_pred = Kff
        for q in range(self.Q):
            # MOGP latent functions prior
            Kuu_q = self.kernels[q].K(self.z[:, :, q], self.z[:, :, q])
            iKuu_q, _ = torch.solve(torch.eye(self.M), Kuu_q)  # is pseudo-inverse?
            Kuu[:, :, q] = Kuu_q
            iKuu[:, :, q] = iKuu_q

            # Variational parameters + Gaussian integration
            q_L = torch.tril(self.q_L[:, :, q])
            q_S[:, :, q] = torch.mm(q_L, q_L.t())

            A = Kfu[:, :, q].mm(iKuu[:, :, q])
            AT = iKuu[:, :, q].mm(Kfu[:, :, q].t())

            m_pred += A.mm(q_m[:, q:q + 1])
            S_pred += A.mm(q_S[:, :, q]).mm(AT) - A.mm(Kfu[:, :, q].t())

        # Detach and numpy easier for plotting.
        m_pred = m_pred.detach().numpy()
        S_pred = S_pred.detach().numpy()

        gp_mu = m_pred.flatten()
        gp_var = np.diagonal(S_pred)

        gp = gp_mu
        gp_upper = gp_mu + 2 * np.sqrt(gp_var)  # + 2*self.likelihood.sigma.detach().numpy()
        gp_lower = gp_mu - 2 * np.sqrt(gp_var)  # - 2*self.likelihood.sigma.detach().numpy()

        return gp, gp_upper, gp_lower

    def rmse(self, x_new, f_new, d):
        f_gp,_,_ = self.predictive(x_new, d)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x_new, f_new, d):
        f_gp,_,_ = self.predictive(x_new, d)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, x_new, y_new, d):
        f_gp, u_gp, _ = self.predictive(x_new, d)
        f_gp = torch.from_numpy(f_gp)
        u_gp = torch.from_numpy(u_gp)
        v_gp = torch.pow(0.5*(u_gp - f_gp), 2.0)
        nlpd = - torch.mean(self.likelihoods[d].log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd
