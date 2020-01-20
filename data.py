import random

import numpy as np
import torch

from utils import *

class RichDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , data
    ):
        super(RichDataset).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    # Гененрирует два случайных семпла, забиваем на индекс
    def __getitem__(self, _):
        idx1 = random.randint(0, self.data.shape[0] - 1)
        idx2 = random.randint(0, self.data.shape[0] - 1)
        return (self.data[idx1], self.data[idx2])

# Хотим разбить на куски: dll + вход + веса, и сгенерить noice
# ->: настоящий выход + вход 1, шум 1 + вход 1, шум 2 + вход 2, веса 1, веса 2
class collate_fn_rich:
    def __init__(self, noise_size):
        self.noise_size = noise_size

    # (arr1, arr2)
    def __call__(self, samples):
        batch_size = len(samples)

        full_1 = torch.cat([torch.tensor(t1).unsqueeze(0) for (t1, t2) in samples], dim=0)
        full_2 = torch.cat([torch.tensor(t2).unsqueeze(0) for (t1, t2) in samples], dim=0)

        input_1 = full_1[:, DLL_DIM:-1]
        input_2 = full_2[:, DLL_DIM:-1]

        w_1 = full_1[:, -1]
        w_2 = full_2[:, -1]

        noise_1 = torch.tensor(np.random.normal(size=(batch_size, self.noise_size))).float()
        noise_2 = torch.tensor(np.random.normal(size=(batch_size, self.noise_size))).float()

        noised_1 = torch.cat((noise_1, input_1), dim=1)
        noised_2 = torch.cat((noise_2, input_2), dim=1)
        real_1 = full_1[:, :-1]

        return (
            real_1.to(device)
            , noised_1.to(device)
            , noised_2.to(device)
            , w_1.to(device)
            , w_2.to(device)
        )
