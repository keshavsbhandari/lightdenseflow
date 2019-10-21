import torch
from utils.warper import warper

def photometricloss(batch, flow, occlusion):
    def robustloss(diff, eps=1e-2, q=4e-1):
        return torch.pow(torch.abs(diff) + eps, q)
    I1 = batch['frame1']
    I2 = batch['frame2']
    _I1 = warper(flow, batch['frame2'])
    occdiff = 1 - occlusion
    num = torch.sum(robustloss(I1 - _I1),1) * occdiff
    deno = torch.sum(occdiff)
    return torch.sum(num/deno)






