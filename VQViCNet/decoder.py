
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 2
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel, stride=stride),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim,
                               kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, 3, kernel_size=kernel,
                               stride=stride)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
