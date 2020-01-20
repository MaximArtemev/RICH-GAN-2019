import torch

DLL_DIM = 5
INPUT_DIM = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
