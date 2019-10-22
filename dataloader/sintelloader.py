from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.getsintelpath import getSintelPairFrame
from utils.masktransform import WDTransformer
from utils.censustransform import censustransform
import torch

USE_CUT_OFF = 10
import random


class SintelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, shape=(256, 256), use_l2=True, channel_in=3, stride=1, kernel_size=2, transform=None,
                 usecst=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pairframe = getSintelPairFrame(root_dir)
        # if USE_CUT_OFF: self.pairframe = random.sample(self.pairframe, USE_CUT_OFF)
        self.transform = transform
        self.usecst = usecst
        self.cst = lambda x: censustransform(x) if self.usecst else x
        self.flow = WDTransformer(shape=shape, use_l2=use_l2, channel_in=channel_in, stride=stride,
                                  kernel_size=kernel_size)

    def __len__(self):
        return 20
        return len(self.pairframe)

    def __getitem__(self, idx):
        # return \
        #     {'frame1':torch.rand(3,256,256),
        #      'frame2':torch.rand(3,256,256),
        #      'displacement':torch.rand(2,256,256),}

        frame1, frame2 = [*map(lambda x: Image.open(x), self.pairframe[idx])]
        sample = {}
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            sample['frame1'] = frame1
            sample['frame2'] = frame2
            sample['displacement'] = self.flow(self.cst(frame1.unsqueeze(0)), self.cst(frame2.unsqueeze(0))).squeeze(
                0).permute(2, 0, 1)

        return sample


class SintelLoader(object):
    def __init__(self, sintel_root="/data/keshav/sintel/training/final", **loaderconfig):
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.ToTensor()])
        self.sinteldataset = SintelDataset(root_dir=sintel_root, transform=self.transform)
        self.loaderconfig = loaderconfig

    def load(self):
        return DataLoader(self.sinteldataset, **self.loaderconfig)
