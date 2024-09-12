#
# This file originates from
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/tree/c50df7816714007c7f2f5188995807b3b396ad3d, licensed
# under the MIT license (see CIR-LICENSE in the root folder of this repository).
#
import torch
import torch.nn as nn
import torch.nn.functional as F

GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)
