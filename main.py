# -*- coding: utf-8 -*-

from typing import Dict

import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()


class NmInput(nn.Module):

    def __init__(self, device):
        super(NmInput, self).__init__()
        self.layer1 = nn.Linear(2048, 512, device=device)
        self.layer2 = nn.Linear(512, 128, device=device)
        self.layer3 = nn.Linear(128, 16, device=device)
        self.layer4 = nn.Linear(16, 1, device=device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == '__main__':
    model = NmInput(torch.device('cpu'))
    input = torch.randn([1024, 2048])
    print(model(input).shape)
    model.eval()
    model = torch.jit.script(model)
    torch.jit.save(model,"testmodel.pt")
