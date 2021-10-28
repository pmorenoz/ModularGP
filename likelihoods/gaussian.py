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

class Gaussian(Likelihood):
    """
    Class for Gaussian Likelihood
    """
    def __init__(self, sigma=None, fit_noise=False):
        super(Gaussian, self).__init__()

        if sigma is None:
            sigma=1.0

        self.sigma = torch.nn.Parameter(sigma*torch.ones(1), requires_grad=fit_noise)


    def pdf(self, f, y):
        normal = Normal(loc=f, scale=self.sigma)
        pdf = torch.exp(normal.log_prob(y))
        return pdf

    def logpdf(self, f, y):
        normal = Normal(loc=f, scale=self.sigma)
        logpdf = normal.log_prob(y)
        return logpdf

    def variational_expectation(self, y, m, v):
        # Variational Expectation of log-likelihood -- Analytical
        lik_variance = self.sigma.pow(2)
        expectation = - np.log(2*np.pi) - torch.log(lik_variance) \
                      - (y.pow(2) + m.pow(2) + v - (2*m*y)).div(lik_variance)

        return 0.5*expectation

    def log_predictive(self, y_test, mu_gp, v_gp, num_samples=1000):
        # function samples:
        normal = Normal(loc=mu_gp.flatten(), scale=torch.sqrt(v_gp).flatten())
        f_samples = normal.sample(sample_shape=(1,num_samples))[0,:,:]

        # monte-carlo:
        logpdf = self.logpdf(f_samples, y_test.flatten())
        log_pred = -np.log(num_samples) + torch.logsumexp(logpdf, dim=0)
        return log_pred