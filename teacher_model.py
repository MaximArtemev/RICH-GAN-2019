import torch
from torch import nn

from utils import *

class TeacherCritic(nn.Module):
    def __init__(
        self
        , hidden_size
        , cramer_size
        , num_layers
    ):
        super(TeacherCritic, self).__init__()

        self.dnn = nn.Sequential(*(
            [nn.Linear(DLL_DIM + INPUT_DIM, hidden_size), nn.SELU()]
            + [nn.Linear(hidden_size, hidden_size), nn.SELU()] * (num_layers - 1)
            + [nn.Linear(hidden_size, cramer_size)]
        ))

    def forward(self, x, y):
        discriminated_x = self.dnn(x)
        discriminated_y = self.dnn(y)

        return (discriminated_x - discriminated_y).norm(dim=1) - discriminated_x.norm(dim=1)


# По правой части таблички генрирует левую
class TeacherGenerator(nn.Module):
    def __init__(
        self
        , noise_size
        , hidden_size
        , num_layers
    ):
        super(TeacherGenerator, self).__init__()

        self.dnn = nn.Sequential(*(
            [nn.Linear(noise_size + INPUT_DIM, hidden_size), nn.SELU()]
            + [nn.Linear(hidden_size, hidden_size), nn.SELU()] * (num_layers - 1)
            + [nn.Linear(hidden_size, DLL_DIM)]
        ))

    def forward(self, noised):
        dll_part = self.dnn(noised)

        return torch.cat((dll_part, noised[:, -INPUT_DIM:]), dim=1)
