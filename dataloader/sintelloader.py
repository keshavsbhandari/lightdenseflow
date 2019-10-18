from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.getsintelpath import getSintelPairFrame
from utils.masktransform import WDTransformer


class SintelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, shape=(256, 256), use_l2=True, channel_in=3, stride=1, kernel_size=2, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pairframe = getSintelPairFrame(root_dir)
        self.transform = transform
        self.flow = WDTransformer(shape=shape, use_l2=use_l2, channel_in=channel_in, stride=stride,
                                  kernel_size=kernel_size)

    def __len__(self):
        return len(self.pairframe)

    def __getitem__(self, idx):
        frame1, frame2 = [*map(lambda x: Image.open(x), self.pairframe[idx])]
        sample = {'frame1': frame1,
                  'frame2': frame2, }
        if self.transform:
            sample['frame1'] = self.transform(sample['frame1'])
            sample['frame2'] = self.transform(sample['frame2'])
            sample['displacement'] = self.flow(sample['frame1'].unsqueeze(0),
                                               sample['frame2'].unsqueeze(0)).squeeze(0).permute(2, 0, 1)
        return sample


class SintelLoader(object):
    def __init__(self, sintel_root="/data/keshav/sintel/training/final", **loaderconfig):
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.ToTensor()])
        self.sinteldataset = SintelDataset(root_dir=sintel_root, transform=self.transform)
        self.loaderconfig = loaderconfig

    def load(self):
        return DataLoader(self.sinteldataset, **self.loaderconfig)