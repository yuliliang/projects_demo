import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yuli_CNN_AE_Resnet_config as cfg
import torchvision.transforms as transforms
device = cfg.device

# ======== data loader ===========
class Dataset4Image(torch.utils.data.Dataset):
    """
    Goal: Process the input files & output image data.
    """
    def __init__(self, DIR_data_file):
        data = pd.read_csv(DIR_data_file)
        data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))
        self.df = data
        self.transform = transforms.Compose([transforms.ToTensor()])

    def img_show(self, idx, noise_sigma=0):
        """ Convert the input data into printable images format. e.g., size = [1, 48, 48]"""
        pixels = self.df['pixels'].iloc[idx]
        pixels = pixels.reshape(cfg.img_h_in, cfg.img_w_in) / 255
        pixels = np.array(pixels, 'float32')
        if noise_sigma != 0:
            noise = np.random.normal(loc=0.0, scale=noise_sigma, size=pixels.shape)
            pixels += noise
            pixels = np.clip(pixels, 0, 1)  # Make sure the data number within the right range.
        return pixels

    def img_tensor(self, pixels):
        """ Convert the input data into tensor images. With batch dimension as 1."""
        if self.transform is not None:
            pixels = self.transform(pixels)
        pixels = pixels.unsqueeze(0)  # insert the batch dimension for VAE
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