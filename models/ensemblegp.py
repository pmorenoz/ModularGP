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
from GPy.inference.latent_function_inference.posterior import Posterior
import numpy as np

class EnsembleGP(torch.nn.Module):
    """
    -- Ensemble Variational Inference for Gaussian Processes --
    """
    def __init__(self, kernel, likelihood, models, M, input_dim=None):
        super(EnsembleGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        # Dimensions --
        self.M = M  # num. inducing
        self.input_dim = int(input_dim)  # dimension of x

        # Ensemble GP Elements --
        self.likelihood = likelihood
        self.kernel = kernel

        if self.input_dim > 1:
            self.z = torch.nn.Parameter(2*torch.rand(self.M, self.input_dim) - 1.0, requires_grad=False)
        else:
            self.z = torch.nn.Parameter(torch.linspace(-0.9, 0.9, self.M)[:,None], requires_grad=False)

        # Adjacent GP Models
        self.models = models  # is a list

        # Ensemble Variational distribution --
        self.q_m = torch.nn.Parameter(torch.randn(M, 1), requires_grad=True)  # variational: mean parameter
        self.q_L = torch.nn.Parameter(torch.eye(M), requires_grad=True)  # variational: covariance

    def ensemble(self):
        # GP prior
        Kuu = self.kernel.K(self.z, self.z)
        iKuu, _ = torch.solve(torch.eye(self.M), Kuu)  # is pseudo-inverse?

        q_m = self.q_m
        q_L = torch.tril(self.q_L)
        q_S = torch.mm(q_L, q_L.t())

        ensemble_m = []
        ensemble_S = []

        # Ensemble GP Distributions
        for model_k in self.models:
            Kkk = self.kernel.K(model_k.z, model_k.z)
            Kuk = self.kernel.K(self.z, model_k.z)
            Kku = torch.transpose(Kuk,0,1)

            A = Kku.mm(iKuu)
            AT = iKuu.mm(Kuk)

            m_k = Kku.mm(iKuu).mm(q_m)
            S_k = Kkk + A.mm(q_S - Kuu).mm(AT)

            ensemble_m.append(m_k)
            ensemble_S.append(S_k)

        return ensemble_m, ensemble_S

    def expectation(self):
        E = 0.0
        ensemble_m, ensemble_S = self.ensemble()

        # Expectation of k ensembles --
        for k,model_k in enumerate(self.models):
            # Ensemble GP -- q_e()
            m_e = ensemble_m[k]
            S_e = ensemble_S[k]

            # Past GP variational distribution -- q_k()
            m_k = model_k.q_m
            L_k = torch.tril(model_k.q_L)
            S_k = torch.mm(L_k, L_k.t())
            iS_k, _ = torch.solve(torch.eye(model_k.M), S_k)  # is pseudo-inverse?

            # Past GP prior -- p_k()
            z_k = model_k.z
            Kkk = model_k.kernel.K(z_k, z_k)
            iKkk, _ = torch.solve(torch.eye(model_k.M), Kkk)  # is pseudo-inverse?

            # Expectation on terms -- E[log_p()] and E[log_q()]
            E_log_q = -torch.trace(iS_k.mm(S_e)) - (m_e - m_k).t().mm(iS_k).mm(m_e - m_k) - torch.logdet(2*np.pi*S_k)
            E_log_p = -torch.trace(iKkk.mm(S_e)) - m_e.t().mm(iKkk).mm(m_e) - torch.logdet(2*np.pi*Kkk)

            # General Expectation -- E[sum_k E[log_q_k] - E[log_p_k]]
            E += 0.5*(E_log_q - E_log_p) + model_k.logZ

        return E

    def divergence(self, p, q):
        kl = kl_divergence(q,p)
        return kl

    def forward(self):

        # Variational parameters --
        q_m = self.q_m
        q_L = torch.tril(self.q_L)
        q_S = torch.mm(q_L, q_L.t())

        # Prior parameters (uses kernel) --
        Kuu = self.kernel.K(self.z, self.z)

        # Distributions -- q(u), p(u)
        q_u = Normal(q_m.flatten(), q_S)
        p_u = Normal(torch.zeros(self.M), Kuu)

        # Expectation --
        expectation = self.expectation()

        # KL divergence --
        kl = self.divergence(q_u, p_u)

        # Calls ELBO
        elbo = expectation - kl
        return -elbo

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
        gp_upper = gp_mu + 2*np.sqrt(gp_var) #+ 2*self.likelihood.sigma.detach().numpy()
        gp_lower = gp_mu - 2*np.sqrt(gp_var) #- 2*self.likelihood.sigma.detach().numpy()

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

