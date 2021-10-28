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

import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior


class SVGP(torch.nn.Module):
    """
    -- Sparse Variational Gaussian Process --
    --
    -- Adaptation to Pytorch + GP framework
    -- Based on Hensman et al. "Scalable Variational Gaussian Process Classification" AISTATS 2015
    -- Reference: http://proceedings.mlr.press/v38/hensman15.pdf
    """
    def __init__(self, kernel, likelihood, M, input_dim=None, batch_rate=1.0):
        super(SVGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        # Dimensions --
        self.M = M                          #num. inducing
        self.input_dim = int(input_dim)     #dimension of x
        self.batch_rate =  batch_rate       #rate of mini-batch/dataset

        # GP Elements --
        self.likelihood = likelihood        #type of likelihood
        self.kernel = kernel                #type of kernel

        self.logZ = 0.0

        if self.input_dim > 1:
            self.z = torch.nn.Parameter(2*torch.rand(self.M, self.input_dim) - 1.0, requires_grad=False)
        else:
            self.z = torch.nn.Parameter(torch.linspace(-0.9, 0.9, self.M)[:,None], requires_grad=False)

        # Variational distribution --
        self.q_m = torch.nn.Parameter(torch.randn(M,1), requires_grad=True)  # variational: mean parameter
        self.q_L = torch.nn.Parameter(torch.eye(M), requires_grad=True)  # variational: covariance

    def forward(self, x, y):

        # Variational parameters --
        q_m = self.q_m
        q_L = torch.tril(self.q_L)
        q_S = torch.mm(q_L, q_L.t())

        # Prior parameters (uses kernel) --
        Kuu = self.kernel.K(self.z)

        # Distributions -- q(u), p(u)
        q_u = Normal(q_m.flatten(), q_S)
        p_u = Normal(torch.zeros(self.M), Kuu)

        # Calculus of q(f) --
        Kff = self.kernel.K(x,x)
        Kfu = self.kernel.K(x, self.z)
        Kuf = torch.transpose(Kfu,0,1)
        iKuu,_ = torch.solve(torch.eye(self.M), Kuu)  # is pseudo-inverse?

        A = Kfu.mm(iKuu)
        AT = iKuu.mm(Kuf)

        m_f = A.mm(q_m)
        v_f = torch.diag(Kff + A.mm(q_S - Kuu).mm(AT))

        # Expectation term --
        expectation = self.likelihood.variational_expectation(y, m_f, v_f)

        # KL divergence --
        kl = kl_divergence(q_u, p_u)

        # Lower bound (ELBO) --
        elbo = self.batch_rate*expectation.sum() - kl
        return -elbo

    def predictive(self, x_new, lik_noise=False):
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
        if lik_noise:
            gp_upper = gp_mu + 2 * np.sqrt(gp_var) + 2 * self.likelihood.sigma.detach().numpy()
            gp_lower = gp_mu - 2 * np.sqrt(gp_var) - 2 * self.likelihood.sigma.detach().numpy()
        else:
            gp_upper = gp_mu + 2*np.sqrt(gp_var)
            gp_lower = gp_mu - 2*np.sqrt(gp_var)

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

    def evidence(self, x, y, N_samples=None):
        # Approximation CI
        if N_samples is None:
            N_samples = 1000

        N,_ = x.shape
        v_f = torch.zeros(N)
        for i in range(N):
            v_f[i] = self.kernel.K(x[i:i+1,:],x[i:i+1,:])
        #v_f = torch.diag(self.kernel.K(x,x), 0)
        m_f = torch.zeros(v_f.shape)
        p_f = Normal(m_f, torch.diag(v_f))
        f_samples = p_f.sample([N_samples]).t()    # N x N_samples
        mc_pdf = self.likelihood.pdf(f_samples, torch.tile(y, (1,N_samples)))

        mc_expectations = 1/N_samples * torch.sum(torch.clamp(mc_pdf, min=1e-100),1)
        print(mc_expectations)
        logZ = torch.sum(torch.log(mc_expectations))

        self.logZ = logZ
        return logZ


