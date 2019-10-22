import torch
from utils.warper import warper


def photometricloss(I, I_, occ, eps=1e-2, q=4e-1):
    error = torch.pow(torch.abs(I - I_) + eps, q) * occ
    occsum = occ.view(occ.size(0), -1).sum(-1).unsqueeze(-1)
    error = error.view(error.size(0), -1) / occsum
    return error.sum()
