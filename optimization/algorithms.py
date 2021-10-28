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
import matplotlib.pyplot as plt

from likelihoods.gaussian import Gaussian
from likelihoods.bernoulli import Bernoulli

class AlgorithmVEM():
    def __init__(self, model, x=None, y=None, config='svgp', iters=20):
        super(AlgorithmVEM, self).__init__()

        self.model = model
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.iters = iters

        if config == 'svgp' or config == 'ensemble':
            self.config = config
        else:
            raise ValueError('Not valid model type for Algorithm VEM, choose \'svgp\' or \'ensemble\'')

        if self.config == 'svgp':
            # Learning rates per param.
            self.lr_m = 1e-6
            self.lr_L = 1e-12
            self.lr_hyp = 1e-10
            self.lr_z = 1e-10

            # VE + VM iterations.
            self.ve_its = 20
            self.vm_its = 10
            self.z_its = 10

        elif self.config == 'ensemble':
            # Learning rates per param.
            self.lr_m = 1e-3
            self.lr_L = 1e-6
            self.lr_hyp = 1e-8
            self.lr_z = 1e-6

            # VE + VM iterations.
            self.ve_its = 30
            self.vm_its = 10
            self.z_its = 10

    def fit(self, opt='sgd', plot=False):
        if opt == 'sgd':
            ve_optimizer = torch.optim.SGD([{'params':self.model.q_m, 'lr':self.lr_m},{'params':self.model.q_L,'lr':self.lr_L}], lr=1e-12, momentum=0.9)

            if isinstance(self.model, Gaussian):
                vm_optimizer = torch.optim.SGD([{'params':self.model.kernel.parameters(), 'lr':self.lr_hyp},{'params':self.model.likelihood.sigma,'lr':self.lr_hyp}], lr=1e-12, momentum=0.9)
            else:
                vm_optimizer = torch.optim.SGD([{'params': self.model.kernel.parameters(), 'lr': self.lr_hyp}], lr=1e-12, momentum=0.9)

            z_optimizer = torch.optim.SGD([{'params':self.model.z, 'lr':self.lr_z}], lr=1e-10, momentum=0.9)

            elbo_its = np.empty((self.iters, 1))
            for em_it in range(self.iters):

                # VE STEP
                for it in range(self.ve_its):
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x,self.y)    # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    ve_optimizer.zero_grad()
                    elbo_it.backward()      # Backward pass <- computes gradients
                    ve_optimizer.step()

                    # Overfitting avoidance
                    if self.config == 'ensemble':
                        if self.model().item() < 10.0:
                            break

                # VM STEP
                # 1. hyper-parameters
                for it in range(self.vm_its):
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x,self.y)    # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    vm_optimizer.zero_grad()
                    elbo_it.backward()      # Backward pass <- computes gradients
                    vm_optimizer.step()

                    # Overfitting avoidance
                    if self.config == 'ensemble':
                        if self.model().item() < 10.0:
                            break

                # 2. inducing-points
                for it in range(self.z_its):
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x,self.y)    # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    z_optimizer.zero_grad()
                    elbo_it.backward()  # Backward pass <- computes gradients
                    z_optimizer.step()

                    # Overfitting avoidance
                    if self.config == 'ensemble':
                        if self.model().item() < 10.0:
                            break

                print('Variational EM step (it=' + str(em_it) + ')')
                if self.config == 'svgp':
                    print('  \__ elbo =', self.model(self.x, self.y).item())
                    elbo_its[em_it] = - self.model(self.x, self.y).item()
                elif self.config == 'ensemble':
                    print('  \__ elbo =', self.model().item())
                    elbo_its[em_it] = - self.model().item()

                    # Overfitting avoidance
                    if self.model().item() < 10.0:
                        break

        elif opt == 'lbfgs':
            optim_param= torch.optim.LBFGS([self.model.q_m, self.model.q_L], lr=self.lr_m, max_iter=self.ve_its)
            optim_hyper = torch.optim.LBFGS(list(self.model.kernel.parameters()) + [self.model.likelihood.sigma], lr=self.lr_hyp, max_iter=self.vm_its)
            optim_z = torch.optim.LBFGS([self.model.z], lr=self.lr_z, max_iter=self.vm_its)

            elbo_its = np.empty((self.iters, 1))
            for em_it in range(self.iters):

                # VE STEP
                def closure():
                    optim_param.zero_grad()
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x, self.y)  # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    elbo_it.backward()
                    return elbo_it

                optim_param.step(closure)
                if self.config == 'svgp':
                    print('  param >>> elbo =', self.model(self.x, self.y).item())
                elif self.config == 'ensemble':
                    print('  param >>> elbo =', self.model().item())

                # VM STEP
                # 1. hyper-parameters
                def closure():
                    optim_hyper.zero_grad()
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x, self.y)  # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    elbo_it.backward()
                    return elbo_it

                optim_hyper.step(closure)
                if self.config == 'svgp':
                    print('  hyper >>> elbo =', self.model(self.x, self.y).item())
                elif self.config == 'ensemble':
                    print('  hyper >>> elbo =', self.model().item())

                # 2. inducing-points
                def closure():
                    optim_z.zero_grad()
                    if self.config == 'svgp':
                        elbo_it = self.model(self.x, self.y)  # Forward pass -> computes ELBO
                    elif self.config == 'ensemble':
                        elbo_it = self.model()  # Forward pass -> computes ELBO

                    elbo_it.backward()
                    return elbo_it

                optim_z.step(closure)
                if self.config == 'svgp':
                    print('  z pts >>> elbo =', self.model(self.x, self.y).item())
                elif self.config == 'ensemble':
                    print('  z pts >>> elbo =', self.model().item())


                print('Variational EM step (it=' + str(em_it) + ')')
                if self.config == 'svgp':
                    print('  \__ elbo =', self.model(self.x, self.y).item())
                    elbo_its[em_it] = - self.model(self.x, self.y).item()
                elif self.config == 'ensemble':
                    print('  \__ elbo =', self.model().item())
                    elbo_its[em_it] = - self.model().item()

        else:
            print('Not valid optimizer')

        if plot:
            plt.figure()
            plt.plot(elbo_its, 'k-')
            plt.title('Ensemble GP Inference (ELBO)')
            plt.xlabel('Iterations')
            plt.show()

def GPR_Optimizer(model, x, y, its=50, lr=1e-2):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=10)
    elbo_its = np.empty((its, 1))
    for it in range(its):
        def closure():
            optimizer.zero_grad()
            elbo_opt = model(x, y)
            elbo_opt.backward()
            return elbo_opt

        optimizer.step(closure)

        print('Optimization step (it=' + str(it) + ')')
        print('  \__ log_marginal =', model(x, y).item())
        elbo_its[it] = -model(x, y).item()


def vem_algorithm(model, x, y, em_iters=10, optimizer='sgd',plot=False):
    if optimizer=='sgd':
        ve_optimizer = torch.optim.SGD([{'params':model.q_m, 'lr':1e-6},{'params':model.q_L,'lr':1e-12}], lr=1e-12, momentum=0.9)
        vm_optimizer = torch.optim.SGD(model.kernel.parameters(), lr=1e-10, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params':model.z, 'lr':1e-10}], lr=1e-10, momentum=0.9)

        VE_iters = 20
        VM_iters = 10
        Z_iters = 10

        elbo_its = np.empty((em_iters, 1))
        for em_it in range(em_iters):

            # VE STEP
            for it in range(VE_iters):
                elbo_it = model(x,y)    # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(VM_iters):
                elbo_it = model(x,y)    # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(Z_iters):
                elbo_it = model(x,y)  # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()  # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model(x, y).item())
            elbo_its[em_it] = -model(x, y).item()


    elif optimizer=='lbfgs':
        ve_optimizer = torch.optim.LBFGS([{model.q_m, model.q_L}], max_iter=50)
        vm_optimizer = torch.optim.LBFGS(model.kernel.parameters(), lr=1e-3, max_iter=10)


        elbo_its = np.empty((em_iters,1))
        for em_it in range(em_iters):
            # VE STEP
            for name, param in model.kernel.named_parameters():
                param.requires_grad = False

                def closure():
                    ve_optimizer.zero_grad()
                    elbo_opt = model(x, y)
                    #print('ELBO:', elbo_opt.item())
                    elbo_opt.backward()
                    return elbo_opt

                ve_optimizer.step(closure)

                # VM STEP
                for name, param in model.kernel.named_parameters():
                    param.requires_grad = True

                    def closure():
                        vm_optimizer.zero_grad()
                        elbo_opt = model(x, y)
                        #print('ELBO:', elbo_opt.item())
                        elbo_opt.backward()
                        return elbo_opt

                vm_optimizer.step(closure)

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model(x, y).item())
            elbo_its[em_it] = -model(x, y).item()

    if plot:
        plt.figure()
        plt.plot(elbo_its, 'k-')
        plt.title('Sparse GP Regression (ELBO)')
        plt.xlabel('Iterations')
        plt.show()

def ensemble_vem(model, em_iters=20, optimizer='sgd',plot=False):
    if optimizer=='sgd':
        ve_optimizer = torch.optim.SGD([{'params':model.q_m, 'lr':1e-3},{'params':model.q_L,'lr':1e-6}], lr=1e-6, momentum=0.9)
        vm_optimizer = torch.optim.SGD(model.kernel.parameters(), lr=1e-8, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params':model.z, 'lr':1e-6}], lr=1e-8, momentum=0.9)

        VE_iters = 30
        VM_iters = 10
        Z_iters = 10

        elbo_its = np.empty((em_iters, 1))
        for em_it in range(em_iters):
            # VE STEP
            # 1. Variational parameters
            for it in range(VE_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(VM_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(Z_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model().item())
            elbo_its[em_it] = -model().item()

            if -model().item() > 0.0:
                break

    if plot:
        plt.figure()
        plt.plot(elbo_its, 'k-')
        plt.title('Ensemble GP Inference (ELBO)')
        plt.xlabel('Iterations')
        plt.show()


def ensemble_vem_parallel(model, em_iters=30, optimizer='sgd',plot=False):
    if optimizer=='sgd':
        ve_optimizer = torch.optim.SGD([{'params':model.q_m, 'lr':1e-3},{'params':model.q_L,'lr':1e-6}], lr=1e-6, momentum=0.9)
        vm_optimizer = torch.optim.SGD(model.kernel.parameters(), lr=1e-8, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params':model.z, 'lr':1e-8}], lr=1e-8, momentum=0.9)

        VE_iters = 30
        VM_iters = 10
        Z_iters = 10

        elbo_its = np.zeros((em_iters, 1))
        for em_it in range(em_iters):
            # VE STEP
            # 1. Variational parameters
            for it in range(VE_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(VM_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(Z_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model().item())
            elbo_its[em_it] = -model().item()

            if -model().item() > 0.0:
                break

    if plot:
        plt.figure()
        plt.plot(elbo_its, 'k-')
        plt.title('Ensemble GP Inference (ELBO)')
        plt.xlabel('Iterations')
        plt.show()

def vem_algorithm_infographic(model, x, y, em_iters=10, plot=False):
    ve_optimizer = torch.optim.SGD([{'params':model.q_m, 'lr':1e-5},{'params':model.q_L,'lr':1e-8}], lr=1e-12, momentum=0.9)
    vm_optimizer = torch.optim.SGD(model.kernel.parameters(), lr=1e-10, momentum=0.9)
    z_optimizer = torch.optim.SGD([{'params':model.z, 'lr':1e-10}], lr=1e-10, momentum=0.9)

    VE_iters = 20
    VM_iters = 20
    Z_iters = 10

    elbo_its = np.empty((em_iters, 1))
    for em_it in range(em_iters):

        # VE STEP
        for it in range(VE_iters):
            elbo_it = model(x,y)    # Forward pass -> computes ELBO
            ve_optimizer.zero_grad()
            elbo_it.backward()      # Backward pass <- computes gradients
            ve_optimizer.step()

        # VM STEP
        # 1. hyper-parameters
        for it in range(VM_iters):
            elbo_it = model(x,y)    # Forward pass -> computes ELBO
            vm_optimizer.zero_grad()
            elbo_it.backward()      # Backward pass <- computes gradients
            vm_optimizer.step()

        # 2. inducing-points
        for it in range(Z_iters):
            elbo_it = model(x,y)  # Forward pass -> computes ELBO
            z_optimizer.zero_grad()
            elbo_it.backward()  # Backward pass <- computes gradients
            z_optimizer.step()

        print('Variational EM step (it=' + str(em_it) + ')')
        print('  \__ elbo =', model(x, y).item())
        elbo_its[em_it] = -model(x, y).item()


def ensemble_vem_infographic(model, em_iters=30, optimizer='sgd',plot=False):
    if optimizer=='sgd':
        ve_optimizer = torch.optim.SGD([{'params':model.q_m, 'lr':1e-3},{'params':model.q_L,'lr':1e-6}], lr=1e-6, momentum=0.9)
        vm_optimizer = torch.optim.SGD(model.kernel.parameters(), lr=1e-8, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params':model.z, 'lr':1e-8}], lr=1e-8, momentum=0.9)

        VE_iters = 30
        VM_iters = 20
        Z_iters = 10

        elbo_its = np.zeros((em_iters, 1))
        for em_it in range(em_iters):
            # VE STEP
            # 1. Variational parameters
            for it in range(VE_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(VM_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(Z_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model().item())
            elbo_its[em_it] = -model().item()

            if -model().item() > 0.0:
                break

def moensemble_vem(model, em_iters=20, optimizer='sgd',plot=False):
    if optimizer=='sgd':
        ve_optimizer = torch.optim.SGD([{'params': model.q_m, 'lr': 1e-3},
                                        {'params': model.q_L,'lr': 1e-6}], lr=1e-6, momentum=0.9)
        vm_optimizer = torch.optim.SGD([{'params': model.kernels.parameters(), 'lr': 1e-8},
                                        {'params': model.coregionalization.W, 'lr': 1e-6}], lr=1e-8, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params': model.z, 'lr':1e-7}], lr=1e-8, momentum=0.9)

        VE_iters = 30
        VM_iters = 20
        Z_iters = 5

        elbo_its = np.empty((em_iters, 1))
        for em_it in range(em_iters):
            # VE STEP
            # 1. Variational parameters
            for it in range(VE_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(VM_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(Z_iters):
                elbo_it = model()    # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()      # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', model().item())
            elbo_its[em_it] = -model().item()

            if -model().item() > 0.0:
                break

    if plot:
        plt.figure()
        plt.plot(elbo_its, 'k-')
        plt.title('Ensemble GP Inference (ELBO)')
        plt.xlabel('Iterations')
        plt.show()

class AlgorithmMOVEM():
    def __init__(self, model, iters=20, plot=False):
        super(AlgorithmMOVEM, self).__init__()

        self.model = model
        self.iters = iters

        # Learning rates per param.
        self.lr_m = 1e-3
        self.lr_L = 1e-6
        self.lr_B = 1e-6
        self.lr_hyp = 1e-8
        self.lr_z = 1e-7

        # VE + VM iterations.
        self.ve_iters = 30
        self.vm_iters = 20
        self.z_iters = 10

    def fit(self, plot=False):

        ve_optimizer = torch.optim.SGD([{'params': self.model.q_m, 'lr': self.lr_m},
                                        {'params': self.model.q_L,'lr': self.lr_L}], lr=1e-6, momentum=0.9)
        vm_optimizer = torch.optim.SGD([{'params': self.model.kernels.parameters(), 'lr': self.lr_hyp},
                                        {'params': self.model.coregionalization.W, 'lr': self.lr_B}], lr=1e-8, momentum=0.9)
        z_optimizer = torch.optim.SGD([{'params': self.model.z, 'lr': self.lr_z}], lr=1e-8, momentum=0.9)

        elbo_its = np.empty((self.iters, 1))
        for em_it in range(self.iters):
            # VE STEP
            # 1. Variational parameters
            for it in range(self.ve_iters):
                elbo_it = self.model()  # Forward pass -> computes ELBO
                ve_optimizer.zero_grad()
                elbo_it.backward()  # Backward pass <- computes gradients
                ve_optimizer.step()

            # VM STEP
            # 1. hyper-parameters
            for it in range(self.vm_iters):
                elbo_it = self.model()  # Forward pass -> computes ELBO
                vm_optimizer.zero_grad()
                elbo_it.backward()  # Backward pass <- computes gradients
                vm_optimizer.step()

            # 2. inducing-points
            for it in range(self.z_iters):
                elbo_it = self.model()  # Forward pass -> computes ELBO
                z_optimizer.zero_grad()
                elbo_it.backward()  # Backward pass <- computes gradients
                z_optimizer.step()

            print('Variational EM step (it=' + str(em_it) + ')')
            print('  \__ elbo =', self.model().item())
            elbo_its[em_it] = -self.model().item()

            if -self.model().item() > 0.0:
                break

        if plot:
            plt.figure()
            plt.plot(elbo_its, 'k-')
            plt.title('Ensemble GP Inference (ELBO)')
            plt.xlabel('Iterations')
            plt.show()