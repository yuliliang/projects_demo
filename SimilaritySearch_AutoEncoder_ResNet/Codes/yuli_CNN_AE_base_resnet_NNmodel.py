import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yuli_CNN_AE_base_resnet_config as cfg
from PIL import Image
import torchvision.transforms as transforms
device = cfg.device


class ConvEncoder(nn.Module):
    """
    Goal: A convolutional Encoder of Auto-Encoder, based on Residual Net structure.
    """
    def __init__(self):
        super().__init__()   # super(ConvEncoder, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.InitialLayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # output [B, 16, (img_H/2)/2, (img_W/2)/2]

        self.maxpool = nn.MaxPool2d((2, 2))   # reduce noise.

        # input image (3, 28, 28)
        # self.ConvLayer1 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=(3,3), stride=1, padding=(1,1)),  # with padding, image shape won't change: (16, 28, 28)
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(16, 16, kernel_size=(3,3), stride=1, padding=(1,1)),  # with padding, image shape won't change: (16, 28, 28)
        #     nn.BatchNorm2d(16),
        #     # nn.MaxPool2d((2, 2))  # (16, 28/2, 28/2)
        # )  # output(16, 14, 14)

        # ..... 0th ResNet basic block (stabilize it first) .....
        self.ConvLayer0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, so image shape won't change: (32, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, image shape won't change: (16, 28, 28)
            nn.BatchNorm2d(64),
            # nn.MaxPool2d((2, 2))  # (32, 14/2, 14/2)
        )  # output(32, 7, 7)
        # self.identity_transform2 = None

        # ..... 1st ResNet basic block .....
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # output [B, 16, (img_H/2), (img_W/2)]

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(128),   # # output [B, 16, img_H, img_W]
        )

        self.ConvLayer1_I_transform = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1), stride=2, padding=(0,0), bias=False),
            nn.BatchNorm2d(128)
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, so image shape won't change: (32, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, image shape won't change: (16, 28, 28)
            nn.BatchNorm2d(128),
            # nn.MaxPool2d((2, 2))  # (32, 14/2, 14/2)
        )  # output(32, 7, 7)
        # self.identity_transform2 = None

        # ..... 2nd ResNet basic block .....
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # output [B, 16, (img_H/2), (img_W/2)]

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(256),   # # output [B, 16, img_H, img_W]
        )

        self.ConvLayer3_I_transform = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1,1), stride=2, padding=(0,0), bias=False),
            nn.BatchNorm2d(256)
        )

        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, so image shape won't change: (32, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, image shape won't change: (16, 28, 28)
            nn.BatchNorm2d(256),
            # nn.MaxPool2d((2, 2))  # (32, 14/2, 14/2)
        )  # output(32, 7, 7)
        # self.identity_transform2 = None

        # ..... 3rd ResNet basic block .....
        self.ConvLayer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=2, padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # output [B, 16, (img_H/2), (img_W/2)]

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(512),   # # output [B, 16, img_H, img_W]
        )

        self.ConvLayer5_I_transform = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,1), stride=2, padding=(0,0), bias=False),
            nn.BatchNorm2d(512)
        )

        self.ConvLayer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, so image shape won't change: (32, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),  # with padding, image shape won't change: (16, 28, 28)
            nn.BatchNorm2d(512),
            # nn.MaxPool2d((2, 2))  # (32, 14/2, 14/2)
        )  # output(32, 7, 7)
        # self.identity_transform2 = None

    def forward(self, x):    #, init_hidden):
        # input x.shape = [B,C,H,W] = [Batch_size, Channel, Height, Width]
        x = self.InitialLayer(x)  # initial conv. with bigger kernel  #in: [B, 3, 48, 32], out: [B, 16, 24, 16]
        x = self.maxpool(x)   # noise clean up  # [B, 16, 12, 8]

        # ..... 0th ResNet basic block .....
        # stabilize the previous denoise signal first
        identity = (x)
        out = self.ConvLayer0(x)   # out.shape = # [B, 32, 6, 4]
        out += identity
        x = self.relu(out)  # x.shape = # [B, 32, 6, 4]

        # ..... 1st ResNet basic block .....
        # increase channel & reduce image size.
        identity = self.ConvLayer1_I_transform(x)  # out: [B, 32, 6, 4]
        out = self.ConvLayer1(x)   # out: [B, 32, 6, 4]
        out += identity   # [B, 32, 6, 4]
        x = self.relu(out)  # [B, 32, 6, 4]

        # same # of channels & same image size.
        identity = (x)
        out = self.ConvLayer2(x)   # out.shape = # [B, 32, 6, 4]
        out += identity
        x = self.relu(out)  # x.shape = # [B, 32, 6, 4]

        # ..... 2nd ResNet basic block .....
        # increase channel & reduce image size.
        identity = self.ConvLayer3_I_transform(x)  # out: [B, 32, 6, 4]
        out = self.ConvLayer3(x)   # out: [B, 32, 6, 4]
        out += identity   # [B, 32, 6, 4]
        x = self.relu(out)  # [B, 32, 6, 4]

        # same # of channels & same image size.
        identity = (x)
        out = self.ConvLayer4(x)   # out.shape = # [B, 32, 6, 4]
        out += identity
        x = self.relu(out)  # x.shape = # [B, 32, 6, 4]

        # ..... 3rd ResNet basic block .....
        # increase channel & reduce image size.
        identity = self.ConvLayer5_I_transform(x)  # out: [B, 32, 6, 4]
        out = self.ConvLayer5(x)   # out: [B, 32, 6, 4]
        out += identity   # [B, 32, 6, 4]
        x = self.relu(out)  # [B, 32, 6, 4]

        # same # of channels & same image size.
        identity = (x)
        out = self.ConvLayer6(x)   # out.shape = # [B, 32, 6, 4]
        out += identity
        x = self.relu(out)  # x.shape = # [B, 32, 6, 4]
        return x      # F.log_softmax(x, dim=1)   # CrossEntropyLoss already have nn.LogSoftmax on it.


class ConvDecoder(nn.Module):
    """
    Goal: A convolutional Decoder of Auto-Encoder, based on Residual Net structure.
    """
    def __init__(self):
        super(ConvDecoder, self).__init__()

        # self.ConvTransposeLayer1 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=1),  # keep the same size
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        # )
        self.relu = nn.ReLU(inplace=True)

        # ..... reverse the 3rd ResNet basic block .....
        self.ConvTransposeLayer6 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )

        self.ConvTransposeLayer5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.ConvTransposeLayer5_I_transform = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, padding=0, bias=False, output_padding=1), # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.),
            nn.BatchNorm2d(256),
        )

        # ..... reverse the 2nd ResNet basic block .....
        self.ConvTransposeLayer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.ConvTransposeLayer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.ConvTransposeLayer3_I_transform = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=1, stride=2, padding=0, bias=False, output_padding=1), # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.),
            nn.BatchNorm2d(128),
        )

        # ..... reverse the 1st ResNet basic block .....
        self.ConvTransposeLayer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.ConvTransposeLayer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.ConvTransposeLayer1_I_transform = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2, padding=0, bias=False, output_padding=1), # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.),
            nn.BatchNorm2d(64),
        )

        # ..... reverse the 0th ResNet basic block .....
        self.ConvTransposeLayer0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # ........
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')  # NOTE: invert max pooling, like unpool

        self.FinishLayer = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):    #, init_hidden):
        # ..... reverse the 3rd ResNet basic block .....
        identity = x    # identity.shape = x.shape = # [B, 32, 6, 4]
        out = self.ConvTransposeLayer6(x)
        out += identity
        x = self.relu(out)

        identity = self.ConvTransposeLayer5_I_transform(x)  # identity.shape = # [B, 16, 12, 8]
        out = self.ConvTransposeLayer5(x)  # out.shape = # [B, 16, 12, 8]
        out += identity
        x = self.relu(out)

        # ..... reverse the 2nd ResNet basic block .....
        identity = x    # identity.shape = x.shape = # [B, 32, 6, 4]
        out = self.ConvTransposeLayer4(x)
        out += identity
        x = self.relu(out)

        identity = self.ConvTransposeLayer3_I_transform(x)  # identity.shape = # [B, 16, 12, 8]
        out = self.ConvTransposeLayer3(x)  # out.shape = # [B, 16, 12, 8]
        out += identity
        x = self.relu(out)

        # ..... reverse the 1st ResNet basic block .....
        identity = x    # identity.shape = x.shape = # [B, 32, 6, 4]
        out = self.ConvTransposeLayer2(x)
        out += identity
        x = self.relu(out)

        identity = self.ConvTransposeLayer1_I_transform(x)  # identity.shape = # [B, 16, 12, 8]
        out = self.ConvTransposeLayer1(x)  # out.shape = # [B, 16, 12, 8]
        out += identity
        x = self.relu(out)

        # ..... reverse the 0st ResNet basic block .....
        identity = x    # identity.shape = x.shape = # [B, 32, 6, 4]
        out = self.ConvTransposeLayer0(x)
        out += identity
        x = self.relu(out)

        # .......
        x = self.unpool(x)  # x.shape = [B, 16, 24, 16]
        x = self.FinishLayer(x)  # x.shape = [B, 3, 48, 32]

        # x = self.ConvTransposeLayer3(x)
        return x


class AutoEncoder(nn.Module):
    """
    Goal: The Auto-Encoder, which connect Encoder & Decoder and output encoder embedding.
    """
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_encoder(self, x):
        x = self.encoder(x)
        return x


# ======== data loader ===========
class Dataset4Image(torch.utils.data.Dataset):
    """
    Goal: The dataset that output image data with input file names (i.e., annotation_file)
    """
    def __init__(self, annotation_file):  # , transform=None):
        # self.data_dir = data_dir
        self.annotations = pd.read_excel(annotation_file)
        del self.annotations['Unnamed: 0']
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize((cfg.img_h_in, cfg.img_w_in))])  # transforms.Resize((1744, 1160))] ) # (Height, Width), int(raw_H/8) * 8,  # make sure it could be integer when compute the finnest embedding.
        # transforms.crop(2, 2, 1744, 1160)   # top: int, left: int, height: int, width:
        # transforms.RandomCrop((299, 299)),
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_FName = self.annotations.iloc[idx, 0]
        img = Image.open(img_FName).convert("RGB")  # img.show()  size = (1166, 1750)= (W, H)
        # y_label = torch.tensor((self.annotations.iloc[idx, 1]))   # .astype(float))
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'img_FName': img_FName}
