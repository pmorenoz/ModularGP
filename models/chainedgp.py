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
from likelihoods.hetgaussian import HetGaussian
from torch.distributions import kl_divergence

import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior


class ChainedGP(torch.nn.Module):
    """
    -- Chained Gaussian Process with Heteroscedastic Gaussian Likelihood --
    --
    -- Adaptation to Pytorch+GP framework
    -- Based on A. Saul et al. "Chained Gaussian Processes" @ AISTATS 2016
    -- Reference: http://proceedings.mlr.press/v51/saul16.pdf
    """
    def __init__(self, kernel_f, kernel_g, M, input_dim=None, batch_rate=1.0):
        super(ChainedGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        # Dimensions --
        self.M = M                          # num. inducing
        self.input_dim = int(input_dim)     # dimension of x
        self.batch_rate = batch_rate        # rate of mini-batch/dataset

        # GP Elements --
        self.likelihood = HetGaussian()     # type of likelihood
        self.kernel_f = kernel_f            # type of kernel for f
        self.kernel_g = kernel_g            # type of kernel for g

        self.logZ = 0.0

        if self.input_dim > 1:
            self.z = torch.nn.Parameter(2*torch.rand(self.M, self.input_dim) - 1.0, requires_grad=False)
        else:
            self.z = torch.nn.Parameter(torch.linspace(-0.9, 0.9, self.M)[:,None], requires_grad=False)

        # Variational distribution f --
        self.q_m_f = torch.nn.Parameter(0.5*torch.randn(M,1), requires_grad=True)  # variational: mean parameter
        self.q_L_f = torch.nn.Parameter(torch.eye(M), requires_grad=True)  # variational: covariance

        # Variational distribution g --
        self.q_m_g = torch.nn.Parameter(0.5*torch.randn(M,1), requires_grad=True)  # variational: mean parameter
        self.q_L_g = torch.nn.Parameter(torch.eye(M), requires_grad=True)  # variational: covariance

    def forward(self, x, y):

        # Variational parameters f --
        q_m_f = self.q_m_f
        q_L_f = torch.tril(self.q_L_f)
        q_S_f = torch.mm(q_L_f, q_L_f.t())

        # Variational parameters g --
        q_m_g = self.q_m_g
        q_L_g = torch.tril(self.q_L_g)
        q_S_g = torch.mm(q_L_g, q_L_g.t())

        # Prior parameters (uses kernel) --
        Kuu_f = self.kernel_f.K(self.z)
        Kuu_g = self.kernel_g.K(self.z)

        # Distributions -- q(u), p(u)
        q_u_f = Normal(q_m_f.flatten(), q_S_f)
        p_u_f = Normal(torch.zeros(self.M), Kuu_f)

        q_u_g = Normal(q_m_g.flatten(), q_S_g)
        p_u_g = Normal(torch.zeros(self.M), Kuu_g)

        # Calculus of q(f) --
        Kff = self.kernel_f.K(x,x)
        Kfu = self.kernel_f.K(x, self.z)
        Kuf = torch.transpose(Kfu,0,1)
        iKuu,_ = torch.solve(torch.eye(self.M), Kuu_f)  # is pseudo-inverse?

        A = Kfu.mm(iKuu)
        AT = iKuu.mm(Kuf)

        m_f = A.mm(q_m_f)
        v_f = torch.diag(Kff + A.mm(q_S_f - Kuu_f).mm(AT))

        # Calculus of q(g) --
        Kff = self.kernel_g.K(x,x)
        Kfu = self.kernel_g.K(x, self.z)
        Kuf = torch.transpose(Kfu,0,1)
        iKuu,_ = torch.solve(torch.eye(self.M), Kuu_g)  # is pseudo-inverse?

        A = Kfu.mm(iKuu)
        AT = iKuu.mm(Kuf)

        m_g = A.mm(q_m_g)
        v_g = torch.diag(Kff + A.mm(q_S_g - Kuu_g).mm(AT))

        # Expectation term --
        expectation = self.likelihood.variational_expectation(y, m_f, v_f, m_g, v_g)

        # KL divergence --
        kl = kl_divergence(q_u_f, p_u_f) + kl_divergence(q_u_g, p_u_g)

        # Lower bound (ELBO) --
        elbo = self.batch_rate*expectation.sum() - kl
        return -elbo

    def predictive(self, x_new, lik_noise=False):
        # Matrices f
        q_m_f = self.q_m_f.detach().numpy()
        q_L_f = torch.tril(self.q_L_f)
        q_S_f = torch.mm(q_L_f, q_L_f.t()).detach().numpy()
        Kuu_f = self.kernel_f.K(self.z, self.z).detach().numpy()

        # Matrices g
        q_m_g = self.q_m_g.detach().numpy()
        q_L_g = torch.tril(self.q_L_g)
        q_S_g = torch.mm(q_L_g, q_L_g.t()).detach().numpy()
        Kuu_g = self.kernel_g.K(self.z, self.z).detach().numpy()

        # GP function f ------
        posterior = Posterior(mean=q_m_f, cov=q_S_f, K=Kuu_f, prior_mean=np.zeros(q_m_f.shape))
        Kx = self.kernel_f.K(self.z, x_new).detach().numpy()
        Kxx = self.kernel_f.K(x_new, x_new).detach().numpy()

        # GP Predictive Posterior - mean + variance
        gp_mu_f = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        gp_var_f = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None, :, :], 1)).T

        gp_f = gp_mu_f
        gp_v_f = gp_var_f

        # GP function g ------
        posterior = Posterior(mean=q_m_g, cov=q_S_g, K=Kuu_g, prior_mean=np.zeros(q_m_g.shape))
        Kx = self.kernel_g.K(self.z, x_new).detach().numpy()
        Kxx = self.kernel_g.K(x_new, x_new).detach().numpy()

        # GP Predictive Posterior - mean + variance
        gp_mu_g = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        gp_var_g = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None, :, :], 1)).T

        gp_g = gp_mu_g
        gp_v_g = gp_var_g

        return gp_f, gp_v_f, gp_g, gp_v_g

    def rmse(self, x_new, f_new):
        f_gp,_,_,_ = self.predictive(x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x_new, f_new):
        f_gp,_,_,_ = self.predictive(x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, x_new, y_new):
        f_gp, v_f_gp, g_gp, v_g_gp = self.predictive(x_new)
        f_gp = torch.from_numpy(f_gp)
        v_f_gp = torch.from_numpy(v_f_gp)
        g_gp = torch.from_numpy(g_gp)
        v_g_gp = torch.from_numpy(v_g_gp)
        nlpd = - torch.mean(self.likelihood.log_predictive(y_new, f_gp, v_f_gp, g_gp, v_g_gp)).detach().numpy()
        return nlpd

    def evidence(self, x, y, N_samples=None):
        # Approximation CI
        if N_samples is None:
            N_samples = 1000

        N,_ = x.shape
        v_f = torch.zeros(N)
        for i in range(N):
            v_f[i] = self.kernel.K(x[i:i+1,:],x[i:i+1,:])

        m_f = torch.zeros(v_f.shape)
        p_f = Normal(m_f, torch.diag(v_f))
        f_samples = p_f.sample([N_samples]).t()    # N x N_samples
        mc_pdf = self.likelihood.pdf(f_samples, torch.tile(y, (1,N_samples)))

        mc_expectations = 1/N_samples * torch.sum(torch.clamp(mc_pdf, min=1e-100),1)
        print(mc_expectations)
        logZ = torch.sum(torch.log(mc_expectations))

        self.logZ = logZ
        return logZ


