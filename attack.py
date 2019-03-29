import numpy as np
import random
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

##############################################################################
# Gradient - Forward pass data through model and return gradient w.r.t. data
##############################################################################
# model - pytorch model to be used for forward pass
# device - device the model is running on
# data -  NCHW tensor of range [0.,1.]
# lbl - label to calculate loss against (usually GT label or target label for targeted attacks)
# returns gradient of loss w.r.t data
# Note: It is important this is treated as an atomic operation because of the
#       normalization. We carry the data around unnormalized, so in this fxn we
#       normalize, forward pass, then unnorm the gradients before returning. Any
#       fxn that uses this method should handle the data in [0,1] range
MEAN = 0.; STD = 1.
def gradient_wrt_data(model,device,data,lbl):
    # MSTAR normalization
    mean = torch.tensor([MEAN], dtype=torch.float32).view([1,1,1]).to(device)
    std = torch.tensor([STD], dtype=torch.float32).view([1,1,1]).to(device)
    # Manually Normalize
    dat = (data-mean)/std
    # Forward pass through the model
    dat.requires_grad = True
    out = model(dat)
    # Calculate loss
    loss = F.cross_entropy(out,lbl)
    # zero all old gradients in the model
    model.zero_grad()
    # Back prop the loss to calculate gradients
    loss.backward()
    # Extract gradient of loss w.r.t data
    data_grad = dat.grad.data
    # Unnorm gradients back into [0,1] space
    #   As shown in foolbox/models/base.py
    grad = data_grad / std
    return grad


##############################################################################
# Projected Gradient Descent Attack (PGD) with random start
##############################################################################
def PGD_attack(model, device, dat, lbl, eps, alpha, iters):
    x_nat = dat.clone().detach()
    # Randomly perturb within small eps-norm ball
    x_adv = dat + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
    # Iteratively Perturb data  
    for i in range(iters):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model,device,x_adv,lbl)
        # Get the sign of the gradient
        sign_data_grad = grad.sign()
        # Perturb by the small amount a
        x_adv = x_adv + alpha*sign_data_grad
        # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
        #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
        x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
        # Make sure we are still in bounds
        x_adv = torch.clamp(x_adv, 0., 1.) 
    return x_adv

##############################################################################
# Projected Gradient Descent Attack (PGD) with random start
##############################################################################
def PGD_attack_test(model, device, dat, lbl, eps, alpha, iters):
    x_nat = dat.clone().detach()
    # Randomly perturb within small eps-norm ball
    x_adv = dat + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
    # Iteratively Perturb data  
    for i in range(iters):

        # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
        #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
        x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
        # Make sure we are still in bounds
        x_adv = torch.clamp(x_adv, 0., 1.) 
    return x_adv

##############################################################################
# FGSM Attack Code - Vanilla
##############################################################################
# Given a model and some data, create an adversarial example using the input 
#    epsilon and return the perturbed data and gradient used to perturb it
def FGSM_attack(model, device, data, lbl, eps):
    dat = data.clone().detach()
    # Forward pass data and get gradient of loss w.r.t data
    grad = gradient_wrt_data(model,device,dat,lbl)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = dat + eps*sign_data_grad
    # Adding clipping to maintain original data range
    perturbed_image = torch.clamp(perturbed_image, 0., 1.)
    return perturbed_image

