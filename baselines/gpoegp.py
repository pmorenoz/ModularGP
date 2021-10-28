# Implementation of the "Generalised Product of GP Experts"
# by Cao & Fleet, 2014
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

class GenPoeGP(torch.nn.Module):
    """
    -- Generalised Product of GP Experts --
    --
    -- Adaptation to Pytorch + GP framework
    -- Y. Cao and D. J. Fleet, "Generalized Product of Experts (...)"
    -- Reference: https://arxiv.org/abs/1410.7827
    """

    def __init__(self, models, input_dim=1.0):
        super(GenPoeGP, self).__init__()

        self.input_dim = int(input_dim)  # dimension of x

        # Adjacent Local GP Models
        self.models = models  # is a list

    def forward(self):
        return 1.0

    def predictive(self, x, y, x_new):
        # x is a list of x_k (distributed)
        # y is a list of y_k (distributed)

        K = len(self.models)
        beta_k = 1/K

        gp_m = torch.zeros(x_new.size())
        gp_v = torch.zeros(x_new.size())

        for k, model_k in enumerate(self.models):
            m_k, v_k = model_k.predictive(x[k], y[k], x_new)

            gp_m += (beta_k*m_k)/v_k
            gp_v += beta_k/v_k

        gp_v = 1.0/gp_v
        gp_m = gp_v*gp_m

        return gp_m, gp_v


    def rmse(self, x, y, x_new, f_new):
        f_gp,_ = self.predictive(x, y, x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x, y, x_new, f_new):
        f_gp,_ = self.predictive(x, y, x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, x, y, x_new, y_new):
        f_gp, v_gp = self.predictive(x, y, x_new)
        nlpd = - torch.mean(self.models[0].likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd

    # FOR HIERARCHICAL SETTINGS

    def predictive_layer(self, gps_m, gps_v, x_new):
        # gps_m is a list of gp_m (distributed)
        # gps_v is a list of gp_v (distributed)

        K = len(gps_m)
        beta_k = 1/K

        gp_m = torch.zeros(x_new.size())
        gp_v = torch.zeros(x_new.size())

        for k, m_k in enumerate(gps_m):
            v_k = gps_v[k]

            gp_m += (beta_k*m_k)/v_k
            gp_v += beta_k/v_k

        gp_v = 1.0/gp_v
        gp_m = gp_v*gp_m

        return gp_m, gp_v

    def rmse_layer(self, gps_m, gps_v, x_new, f_new):
        f_gp,_ = self.predictive_layer(gps_m, gps_v, x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae_layer(self, gps_m, gps_v, x_new, f_new):
        f_gp,_ = self.predictive_layer(gps_m, gps_v, x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd_layer(self, gps_m, gps_v, x_new, y_new):
        f_gp, v_gp = self.predictive_layer(gps_m, gps_v, x_new)
        nlpd = - torch.mean(self.models[0].likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd
