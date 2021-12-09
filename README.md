# Modular Gaussian Processes<br> for Transfer Learning

<img src="/extra/modular_gp_logo.png" width=1000>

## 🧩 Introduction

This repository contains the implementation of our paper [Modular Gaussian Processes for Transfer Learning](https://arxiv.org/abs/2110.13515) accepted in the 35th Conference on Neural Information Processing Systems (NeurIPS) 2021. The entire code is written in Python and is based on the [Pytorch](https://pytorch.org/) framework.

### 🧩 Idea

Here, you may find a new framework for transfer learning based on *modular Gaussian processes* (GP). The underlying idea is to avoid the revisiting of samples once a model is trained and well-fitted, so the model can be repurposed in combination with other or new data. We build *dictionaries* of modules (models), where each one contains only parameters and hyperparameters, but not observations. Finally, we are able to build *meta-models* (GP models) from different combinations of modules without reusing the old data.

## 🧩 Citation

Please, if you use this code, include the following citation:
```
@inproceedings{MorenoArtesAlvarez21,
  title =  {Modular {G}aussian Processes for Transfer Learning},
  author =   {Moreno-Mu\~noz, Pablo and Art\'es-Rodr\'iguez, Antonio and \'Alvarez, Mauricio A},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year =   {2021}
}
```
