
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
from likelihoods.likelihood import Likelihood
from torch.distributions.normal import Normal
from util import safe_exp, safe_square

class HetGaussian(Likelihood):
    """
    Class for Heteroscedastic Gaussian Likelihood
        --
    -- Adaptation to Pytorch+GP framework
    -- Based on M. Lázaro-Gredilla et al. "Variational Heteroscedastic Gaussian Process Regression" @ ICML 2011
    -- Reference: https://icml.cc/Conferences/2011/papers/456_icmlpaper.pdf
    """
    def __init__(self):
        super(HetGaussian, self).__init__()

    def pdf(self, f, g, y):
        normal = Normal(loc=f, scale=safe_exp(g))
        pdf = safe_exp(normal.log_prob(y))
        return pdf

    def logpdf(self, f, g, y):
        normal = Normal(loc=f, scale=safe_exp(g))
        logpdf = normal.log_prob(y)
        return logpdf

    def variational_expectation(self, y, m_f, v_f, m_g, v_g):
        # Variational Expectation of log-likelihood -- Analytical
        precision = torch.clamp(safe_exp(-m_g + (0.5*v_g)), min=-1e9, max=1e9)
        #squares = torch.clamp(safe_square(y) + safe_square(m_f) + v_f - (2*m_f*y), min=-1e9, max=1e9)
        squares = torch.clamp(y**2 + m_f**2 + v_f - (2 * m_f * y), min=-1e9, max=1e9)
        expectation = -np.log(2*np.pi) - m_g - (precision*squares)
        return 0.5*expectation

    def log_predictive(self, y_test, mu_f_gp, v_f_gp, mu_g_gp, v_g_gp, num_samples=1000):
        # function samples f:
        normal = Normal(loc=mu_f_gp.flatten(), scale=torch.sqrt(v_f_gp).flatten())
        f_samples = normal.sample(sample_shape=(1,num_samples))[0,:,:]

        # function samples g:
        normal = Normal(loc=mu_g_gp.flatten(), scale=torch.sqrt(v_g_gp).flatten())
        g_samples = normal.sample(sample_shape=(1,num_samples))[0,:,:]

        # monte-carlo:
        logpdf = self.logpdf(f_samples, g_samples, y_test.flatten())
        log_pred = -np.log(num_samples) + torch.logsumexp(logpdf, dim=0)
        return log_pred