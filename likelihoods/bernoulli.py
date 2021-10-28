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
from torch.distributions.bernoulli import Bernoulli as Ber

class Bernoulli(Likelihood):
    """
    Class for Gaussian Likelihood
    """
    def __init__(self):
        super(Bernoulli, self).__init__()


    def pdf(self, f, y):

        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(f)#.flatten()
        bernoulli = Ber(probs=p)
        pdf = torch.exp(bernoulli.log_prob(y))
        return pdf

    def logpdf(self, f, y):
        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(f).flatten()
        bernoulli = Ber(probs=p)
        logpdf = bernoulli.log_prob(y)
        return logpdf

    def variational_expectation(self, y, m, v):
        # Gauss-Hermite Quadrature
        gh_p, gh_w = self.gh_points()
        gh_w = torch.div(gh_w, np.sqrt(np.pi))

        m, v, y = m.flatten(), v.flatten(), y.flatten()
        f = gh_p[None, :] * torch.sqrt(2. * v[:, None]) + m[:, None]
        y = y[:,None].repeat(1,f.size(1))

        logp = self.logpdf(f.view(-1), y.view(-1))
        logp = logp.view(f.size()).double()
        gh_w = gh_w[:, None]

        var_exp = logp.mm(gh_w)
        return var_exp

    def log_predictive(self, y_test, mu_gp, v_gp, num_samples=1000):
        N_test = y_test.size(0)
        # function samples:
        normal = Normal(loc=mu_gp.flatten(), scale=torch.sqrt(v_gp).flatten())
        f_samples = torch.reshape(normal.sample(sample_shape=(1,num_samples))[0,:,:], (-1,))

        # monte-carlo:
        logpdf = self.logpdf(f_samples, y_test.repeat(num_samples,1).flatten())
        log_pred = -np.log(num_samples) + torch.logsumexp(logpdf, dim=0)
        return -log_pred

