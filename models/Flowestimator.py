import torch.nn as nn
from torch.nn.functional import tanh
from models.DenseNet import DenseNet
from models.Occlusion import Occlusion


class FlowEstimator(nn.Module):
    def __init__(self):
        super(FlowEstimator, self).__init__()
        self.dense = DenseNet(input_features=2, growth_rate=2, block_config=(8, 8, 8, 8, 8), compression=0.5,
                              num_init_features=8, bn_size=1, drop_rate=0.5, small_inputs=False, efficient=True)
        self.conv2d = nn.Conv2d(kernel_size=3, stride=1, in_channels=31, padding=1, out_channels=2)
        self.occlusion = Occlusion()

    def forward(self, x):
        x = tanh(self.dense(x))
        flow = tanh(self.conv2d(x))
        occ = self.occlusion(flow)
        return flow, occ
