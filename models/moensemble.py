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
from GPy.inference.latent_function_inference.posterior import Posterior
import numpy as np

class MultiOutputEnsembleGP(torch.nn.Module):
    """
    -- Multi Output Ensemble for Gaussian Processes --
    """

    def __init__(self, models, kernels, Q, M, input_dim=None):
        super(MultiOutputEnsembleGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        # Dimensions --
        self.M = M  # num. inducing
        self.K = len(models)  # num. models
        self.input_dim = int(input_dim)  # dimension of x

        # Multi-output GP Ensemble Elements --
        self.Q = Q

        # Kernels --
        self.kernels = torch.nn.ModuleList()
        for q in range(self.Q):
            self.kernels.append(kernels[q])
        self.coregionalization = LMC(self.kernels, self.K)  # is a list

        if self.input_dim > 1:
            self.z = torch.nn.Parameter(torch.rand(self.M, self.input_dim, self.Q), requires_grad=False)
        else:
            self.z = torch.nn.Parameter(torch.tile(torch.linspace(0.1, 0.9, self.M)[:,None, None], (1, 1, self.Q)), requires_grad=False)

        # Adjacent GP Models
        self.models = models  # is a list

        # Ensemble Variational distribution --
        self.q_m = torch.nn.Parameter(2*torch.randn(M, Q), requires_grad=True)  # variational: mean parameter
        self.q_L = torch.nn.Parameter(0.5*torch.tile(torch.eye(M)[:,:,None], (1, 1, self.Q)), requires_grad=True)  # variational: covariance


    def ensemble(self):
        # MOGP prior + Variational parameters
        q_m = self.q_m
        q_S = torch.zeros(self.M, self.M, self.Q)
        Kvv = torch.zeros(self.M, self.M, self.Q)
        iKvv = torch.zeros(self.M, self.M, self.Q)
        for q in range(self.Q):
            Kvv_q = self.kernels[q].K(self.z[:,:,q], self.z[:,:,q])
            iKvv_q, _ = torch.solve(torch.eye(self.M), Kvv_q)  # is pseudo-inverse?
            Kvv[:,:,q] = Kvv_q
            iKvv[:,:,q] = iKvv_q

            q_L = torch.tril(self.q_L[:,:,q])
            q_S[:,:,q] = torch.mm(q_L, q_L.t())

        ensemble_m = []
        ensemble_S = []

        # Ensemble MOGP Distributions
        for k, model_k in enumerate(self.models):

            Kuu = self.coregionalization.Kff(model_k.z, k)
            Kuv = self.coregionalization.Kfu(model_k.z, self.z, k)

            m_k = 0.0
            S_k = Kuu

            for q in range(self.Q):

                A = Kuv[:,:,q].mm(iKvv[:,:,q])
                AT = iKvv[:,:,q].mm(Kuv[:,:,q].t())

                m_k += A.mm(q_m[:,q:q+1])
                S_k += A.mm(q_S[:,:,q]).mm(AT) - A.mm(Kuv[:,:,q].t())

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

    def divergence(self, p_v, q_v):
        kl = 0.0
        for q in range(self.Q):
            kl += kl_divergence(q_v[q], p_v[q])
        return kl

    def forward(self):

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

        # Expectation --
        expectation = self.expectation()

        # KL divergence --
        kl = self.divergence(q_u, p_u)

        # Calls ELBO
        elbo = expectation - kl
        return -elbo

    def predictive(self, xnew, k):
        # MOGP prior + Variational parameters
        q_m = self.q_m
        q_S = torch.zeros(self.M, self.M, self.Q)
        Kvv = torch.zeros(self.M, self.M, self.Q)
        iKvv = torch.zeros(self.M, self.M, self.Q)

        # Posterior distribution on new input data
        Kuu = self.coregionalization.Kff(xnew, k)
        Kuv = self.coregionalization.Kfu(xnew, self.z, k)

        m_k = 0.0
        S_k = Kuu
        for q in range(self.Q):
            # MOGP latent functions prior
            Kvv_q = self.kernels[q].K(self.z[:, :, q], self.z[:, :, q])
            iKvv_q, _ = torch.solve(torch.eye(self.M), Kvv_q)  # is pseudo-inverse?
            Kvv[:, :, q] = Kvv_q
            iKvv[:, :, q] = iKvv_q

            # Variational parameters + Gaussian integration
            q_L = torch.tril(self.q_L[:, :, q])
            q_S[:, :, q] = torch.mm(q_L, q_L.t())

            A = Kuv[:, :, q].mm(iKvv[:, :, q])
            AT = iKvv[:, :, q].mm(Kuv[:, :, q].t())

            m_k += A.mm(q_m[:, q:q + 1])
            S_k += A.mm(q_S[:, :, q]).mm(AT) - A.mm(Kuv[:, :, q].t())

        m_k = m_k.detach().numpy()
        S_k = S_k.detach().numpy()

        gp_mu = m_k.flatten()
        gp_var = np.diagonal(S_k)

        gp = gp_mu
        gp_upper = gp_mu + 2 * np.sqrt(gp_var)  # + 2*self.likelihood.sigma.detach().numpy()
        gp_lower = gp_mu - 2 * np.sqrt(gp_var)  # - 2*self.likelihood.sigma.detach().numpy()

        return gp, gp_upper, gp_lower

    def rmse(self, x_new, f_new, k):
        f_gp,_,_ = self.predictive(x_new, k)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x_new, f_new, k):
        f_gp,_,_ = self.predictive(x_new, k)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, likelihood, x_new, y_new, k):
        f_gp, u_gp, _ = self.predictive(x_new, k)
        f_gp = torch.from_numpy(f_gp)
        u_gp = torch.from_numpy(u_gp)
        v_gp = torch.pow(0.5*(u_gp - f_gp), 2.0)
        nlpd = - torch.mean(likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd