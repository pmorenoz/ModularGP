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

# -----------------------------------------------------------------
# Experiment -- Solar Dataset
# -----------------------------------------------------------------

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.svgp import SVGP
from models.ensemblegp import EnsembleGP
from baselines.distgp import DistGP
from baselines.poegp import PoeGP
from baselines.gpoegp import GenPoeGP
from baselines.bcm import BayesianCM
from baselines.rbcm import RobustBayesianCM
from baselines.dvigp import DVIGP
from optimization.algorithms import AlgorithmVEM
from optimization.algorithms import GPR_Optimizer
from optimization.algorithms import AlgorithmVEM
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# COOLORS.CO palettes
color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']
color_palette_6 = ['#21295c','#1b3b6f','#065a82','#1c7293','#9eb3c2']
color_palette_7 = ['#f7b267','#f79d65','#f4845f','#f27059','#f25c54']

palette = color_palette_4

trials = 10
experiment = 'solar'

recy_metrics = np.zeros((3,trials))
poe_metrics = np.zeros((3,trials))
gpoe_metrics = np.zeros((3,trials))
bcm_metrics = np.zeros((3,trials))
rbcm_metrics = np.zeros((3,trials))

# Load Solar Data --
data = sio.loadmat('../data/nasa.mat')
y = data['nasa'][:,2]
y = np.log(y + 1)
y = y[:,np.newaxis]
y = (y - np.mean(y))    # mean normalization
x = np.linspace(0,100, y.shape[0])[:,np.newaxis]

print(y.shape)


for trial in range(trials):

    print('TRIAL = ' + str(trial) + '/' + str(trials))

    ###########################
    #                         #
    #    DISTRIBUTED TASKS    #
    #                         #
    ###########################

    tasks = 50
    min_x = 0.0
    max_x = 100.0
    segment_x = (max_x - min_x)/tasks
    x_tasks = []    # training x -- inputs
    y_tasks = []    # training y -- outputs

    x_test = torch.zeros(1,1)     # test x -- inputs
    y_test = torch.zeros(1,1)     # test y -- outputs

    n_training = 0
    n_test = 0
    for k in range(tasks):
        min_x_k = min_x + (k*segment_x)
        max_x_k = min_x + ((k+1)*segment_x)
        y_k = y[(x[:, 0] > min_x_k) & (x[:, 0] < max_x_k), :]
        x_k = x[(x[:, 0] > min_x_k) & (x[:, 0] < max_x_k), :]

        x_k_train, x_k_test, y_k_train, y_k_test = train_test_split(x_k, y_k, test_size = 0.2, random_state = 42)

        x_tasks.append(torch.from_numpy(x_k_train).float())
        y_tasks.append(torch.from_numpy(y_k_train).float())

        x_test = torch.cat((x_test, torch.from_numpy(x_k_test).float()), 0)
        y_test = torch.cat((y_test, torch.from_numpy(y_k_test).float()), 0)

        #x_k_test = x_k[::5, :]
        #y_k_test = y_k[::5, :]

        #x_tasks.append(torch.from_numpy(np.delete(x_k,np.s_[::5])[:,None]).float())
        #y_tasks.append(torch.from_numpy(np.delete(y_k,np.s_[::5])[:,None]).float())

        #x_test = torch.cat((x_test, torch.from_numpy(x_k_test).float()), 0)
        #y_test = torch.cat((y_test, torch.from_numpy(y_k_test).float()), 0)

        n_training += y_k_train.shape[0]
        n_test += y_k_test.shape[0]


    print('Total # of tasks: ', len(x_tasks))
    print('Number # of training samples: ', n_training)
    print('Number # of test samples: ', n_test)

    ###########################
    #                         #
    #   PARALLEL INFERENCE    #
    #                         #
    ###########################

    M_k = 6
    models = []       # for recyclable GPs
    models_dist = []  # for distributed GPs
    x_all = []        # for distributed GPs
    y_all = []        # for distributed GPs
    for k, x_k in enumerate(x_tasks):
        print('-                             -')
        print('----- TASK k=' + str(k + 1) + ' ------')
        print('-                             -')
        ######################################################
        # 1. RECYCLABLE GP
        ######################################################
        kernel_k = RBF(length_scale=0.2, variance=1.0)
        likelihood_k = Gaussian(sigma=0.1, fit_noise=True)
        model_k = SVGP(kernel_k, likelihood_k, M_k)

        z_k_min = min_x + (k*segment_x)
        z_k_max = min_x + ((k+1)*segment_x)
        model_k.z = torch.nn.Parameter(torch.linspace(z_k_min, z_k_max, M_k)[:, None], requires_grad=True)

        vem_algorithm = AlgorithmVEM(model_k, x_k, y_tasks[k], iters=20)

        vem_algorithm.ve_its = 20
        vem_algorithm.vm_its = 20
        vem_algorithm.lr_m = 1e-5
        vem_algorithm.lr_L = 1e-8
        vem_algorithm.lr_hyp = 1e-10
        vem_algorithm.lr_z = 1e-10

        vem_algorithm.fit()

        ######################################################
        # 2. DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
        ######################################################

        kernel_j = RBF()
        likelihood_j = Gaussian(fit_noise=False)
        model_j = DistGP(kernel_j, likelihood_j)
        GPR_Optimizer(model_j, x_k, y_tasks[k])

        models_dist.append(model_j)
        x_all.append(x_k)
        y_all.append(y_tasks[k])

    ###########################
    #                         #
    #   ENSEMBLE INFERENCE    #
    #                         #
    ###########################
    print('-                   -')
    print('----- ENSEMBLE ------')
    print('-                   -')

    ######################################################
    # 1. RECYCLABLE GP
    ######################################################

    M_e = 90
    kernel = RBF()
    likelihood = Gaussian(fit_noise=False)
    model_e = EnsembleGP(kernel, likelihood, models, M_e)
    model_e.z = torch.nn.Parameter(torch.linspace(min_x, max_x, M_e)[:, None], requires_grad=True)
    vem_algorithm = AlgorithmVEM(model_e, config='ensemble', iters=10)

    vem_algorithm.ve_its = 30
    vem_algorithm.vm_its = 10
    vem_algorithm.lr_m = 1e-3
    vem_algorithm.lr_L = 1e-6
    vem_algorithm.lr_hyp = 1e-8
    vem_algorithm.lr_z = 1e-8

    vem_algorithm.fit()

    nlpd = model_e.nlpd(x_test, y_test)
    rmse = model_e.rmse(x_test, y_test)
    mae = model_e.mae(x_test, y_test)

    recy_metrics[0, trial] = nlpd
    recy_metrics[1, trial] = rmse
    recy_metrics[2, trial] = mae

    print('Recyclable - NLPD: ', nlpd)
    print('Recyclable - RMSE: ', rmse)
    print('Recyclable - MAE: ', mae)
    print(' ')

    ######################################################
    # 2. DISTRIBUTED GP (FOR BCM, RBCM, POE & GPOE)
    ######################################################

    # A. POE  _________//

    poe_model = PoeGP(models_dist)

    nlpd = poe_model.nlpd(x_all, y_all, x_test, y_test)
    rmse = poe_model.rmse(x_all, y_all, x_test, y_test)
    mae = poe_model.mae(x_all, y_all, x_test, y_test)

    poe_metrics[0, trial] = nlpd
    poe_metrics[1, trial] = rmse
    poe_metrics[2, trial] = mae

    print('POE-NLPD: ', nlpd)
    print('POE-RMSE: ', rmse)
    print('POE-MAE: ',  mae)
    print(' ')

    # B. GPOE _________//

    gpoe_model = GenPoeGP(models_dist)

    nlpd = gpoe_model.nlpd(x_all, y_all, x_test, y_test)
    rmse = gpoe_model.rmse(x_all, y_all, x_test, y_test)
    mae = gpoe_model.mae(x_all, y_all, x_test, y_test)

    gpoe_metrics[0, trial] = nlpd
    gpoe_metrics[1, trial] = rmse
    gpoe_metrics[2, trial] = mae

    print('GenPOE-NLPD: ', nlpd)
    print('GenPOE-RMSE: ', rmse)
    print('GenPOE-MAE: ',  mae)
    print(' ')

    # C. BCM  _________//

    bcm_model = BayesianCM(models_dist)

    nlpd = bcm_model.nlpd(x_all, y_all, x_test, y_test)
    rmse = bcm_model.rmse(x_all, y_all, x_test, y_test)
    mae = bcm_model.mae(x_all, y_all, x_test, y_test)

    bcm_metrics[0, trial] = nlpd
    bcm_metrics[1, trial] = rmse
    bcm_metrics[2, trial] = mae

    print('BCM-NLPD: ', nlpd)
    print('BCM-RMSE: ', rmse)
    print('BCM-MAE: ',  mae)
    print(' ')

    # D. RBCM _________//

    rbcm_model = RobustBayesianCM(models_dist)

    nlpd = rbcm_model.nlpd(x_all, y_all, x_test, y_test)
    rmse = rbcm_model.rmse(x_all, y_all, x_test, y_test)
    mae = rbcm_model.mae(x_all, y_all, x_test, y_test)

    rbcm_metrics[0, trial] = nlpd
    rbcm_metrics[1, trial] = rmse
    rbcm_metrics[2, trial] = mae

    print('RBCM-NLPD: ', nlpd)
    print('RBCM-RMSE: ', rmse)
    print('RBCM-MAE: ',  mae)
    print(' ')

    # save to csv file
    np.savetxt('./metrics/recy_metrics_' + experiment + '.csv', recy_metrics, delimiter=',')
    np.savetxt('./metrics/poe_metrics_' + experiment + '.csv', poe_metrics, delimiter=',')
    np.savetxt('./metrics/gpoe_metrics_' + experiment + '.csv', gpoe_metrics, delimiter=',')
    np.savetxt('./metrics/bcm_metrics_' + experiment + '.csv', bcm_metrics, delimiter=',')
    np.savetxt('./metrics/rbcm_metrics_' + experiment + '.csv', rbcm_metrics, delimiter=',')