import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yuli_CNN_VAE_base_resnet_config as cfg
import torchvision.transforms as transforms
device = cfg.device

# ======== data loader ===========
class Dataset4Image(torch.utils.data.Dataset):
    """
    Goal: The dataset that output image data with input file names (i.e., annotation_file)
    """
    def __init__(self, DIR_data_file):  # , transform=None):
        data = pd.read_csv(DIR_data_file)
        data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))
        self.df = data
        # Converting pixels into numpy array
        self.transform = transforms.Compose(
            [transforms.ToTensor()])
             # transforms.Normalize((0.5), (0.5))])

    def img_show(self, idx):
        """ Convert the input data into printable images format. e.g., size = [1, 48, 48]"""
        pixels = self.df['pixels'].iloc[idx]
        pixels = pixels.reshape(cfg.img_h_in, cfg.img_w_in) / 255
        pixels = np.array(pixels, 'float32')
        return pixels

    def img_tensor(self, idx):
        """ Convert the input data into tensor images format. With batch dimension as 1."""
        pixels = self.img_show(idx)
        if self.transform is not None:
            pixels = self.transform(pixels)   # [1, 48, 48]
        pixels = pixels.unsqueeze(0)  # [1, 1, 48, 48]
        return pixels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df['img_name'].iloc[idx]

        pixels = self.df['pixels'].iloc[idx]
        pixels = pixels.reshape(cfg.img_h_in, cfg.img_w_in) / 255
        pixels = np.array(pixels, 'float32')
        if self.transform is not None:   #
            pixels = self.transform(pixels)

        return {'image': pixels, 'idx': idx, 'img_name': img_name}