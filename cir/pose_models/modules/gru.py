#
# This file originates from
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/tree/c50df7816714007c7f2f5188995807b3b396ad3d, licensed
# under the MIT license (see CIR-LICENSE in the root folder of this repository).
#
import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super().__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        z = torch.sigmoid(self.convz(net_inp))
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)))

        net = (1-z) * net + z * q
        return net
