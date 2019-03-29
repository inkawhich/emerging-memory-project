
import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor):
        E = tensor.abs().mean()        
        return tensor.sign() * E

 
an=2 # ACTIVATION BITS

class Quantizer(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        n = float(2 ** an - 1) 
        input = input*n
        return input.round() / n 

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        return grad_input

class Quantizer_nonlinear(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        #list = [-1,-1/2,-1/4,-1/8,0,1/8,1/4,1/2,1]
        
        
        #list=[-1,-3/4,-1/2,-3/8,-1/4,-3/16,-1/8,-1/16,0,1/16,1/8,3/16,1/4,3/8,1/2,3/4,1]
        list = [-1,-1/3,0,1/3,1]
        for i,num in enumerate(list[:-1]):
            #print(i)
            flag = torch.zeros_like(input)
            quan_res = torch.ones_like(input)*num
            if i == 0:
                torch.where(input/torch.max(torch.abs(input))<=(num+list[i+1])/2,quan_res,input)
            else:
                torch.where((input/torch.max(torch.abs(input))>(num+list[i-1])/2)&(input/torch.max(torch.abs(input))<=(num+list[i+1])/2),quan_res,input)
            if i == len(list)-2:
                quan_res = torch.ones_like(input)*list[i+1]
                torch.where(input/torch.max(torch.abs(input))>(num+list[i+1])/2,quan_res,input)
        return input
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        return grad_input
