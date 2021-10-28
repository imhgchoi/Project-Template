
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class SampleModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.fc = nn.Linear(1024*3, 10)

        self.init_weights(method='none')

    def forward(self, x):
        return F.softmax(self.fc(x.flatten(-3)), -1)