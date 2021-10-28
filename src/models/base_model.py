

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, method):
        if method == 'none':
            pass
        else :
            raise NotImplementedError(f'{method} initialization not implemented')

    def forward(self, x):
        raise NotImplementedError(f'forward method not implemented')