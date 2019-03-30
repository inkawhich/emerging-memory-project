
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


class Quantizer(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self,nbits):
        super(Quantizer,self).__init__()
        self.nbits = nbits
    @staticmethod
    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        n = float(2 ** self.nbits - 1) 
        input = input*n
        return input.round() / n 


    @staticmethod
    def backward(self,grad_output):
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
    def __init__(self,nbits):
        super(Quantizer_nonlinear,self).__init__()
        self.nbits=nbits
    @staticmethod
    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        n = float(2 ** self.nbits - 1)
        input_norm = input/torch.max(torch.abs(input))
        input_norm = torch.sign(input_norm)*1/np.log(1+n)*log(1+n*torch.abs(input_norm))
        
        input = input_norm*torch.max(torch.abs(input))*n
        return input.round() / n
    @staticmethod
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        return grad_input
