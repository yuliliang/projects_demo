import torch
import torch.nn as nn
import yuli_CNN_VAE_Resnet_config as cfg
device = cfg.device


class Encoder_ResNetBlock(nn.Module):
    """The basic ResNet block architecture for Encoder """
    def __init__( self, flag: str, h_dim_in: int, h_dim_out: int) -> None:
        super().__init__()
        self.flag = flag
        self.relu = nn.ReLU(inplace=True)

        if self.flag == 'Residual_DownSample':  # increase hidden dimension & down-sample image size
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(h_dim_in, h_dim_out, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
                nn.BatchNorm2d(h_dim_out),
                nn.ReLU(inplace=True),  # output [B, h_dim_out, (img_H/2), (img_W/2)]

                nn.Conv2d(h_dim_out, h_dim_out, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(h_dim_out),
            )
            self.ConvLayer_I_transform = nn.Sequential(
                nn.Conv2d(h_dim_in, h_dim_out, kernel_size=(1, 1), stride=2, padding=(0, 0), bias=False),
                nn.BatchNorm2d(h_dim_out)  # output [B, h_dim_out, (img_H/2), (img_W/2)]
            )
        elif self.flag == 'Residual_Stable':
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(h_dim_in, h_dim_out, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(h_dim_out),
                nn.ReLU(inplace=True),  # output [B, h_dim_out, (img_H/2), (img_W/2)]

                nn.Conv2d(h_dim_out, h_dim_out, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
                nn.BatchNorm2d(h_dim_out),  # output [B, h_dim_out, (img_H), (img_W)]
            )
        else:
            print("Error: Unknown Encoder_ResNetBlock flag")

    def forward(self, x):
        if self.flag == 'Residual_DownSample':   # increase channel & reduce image size.
            identity = self.ConvLayer_I_transform(x)
        elif self.flag == 'Residual_Stable':    # same # of channels & same image size.
            identity = x
        else:
            print("Error: Unknown Encoder_ResNetBlock flag")
        out = self.ConvLayer(x)
        out += identity
        x = self.relu(out)
        return x


class ConvEncoder(nn.Module):
    """
    Goal: A convolutional Encoder of Auto-Encoder, based on Residual Net structure.
    """
    def __init__(self, hidden_dims, flattened_channels, latent_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.relu = nn.ReLU(inplace=True)

        # ...... first layer feature extraction .....
        # initial input: [B, cfg.raw_img_channel, (img_H/2), (img_W/2)]
        self.InitialLayer = nn.Sequential(
            nn.Conv2d(cfg.raw_img_channel, hidden_dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d((2, 2))

        # ..... ResNet blocks .....
        self.Encoder_ResidualBlocks = nn.Sequential(
            Encoder_ResNetBlock('Residual_Stable', hidden_dims[0], hidden_dims[0]),

            Encoder_ResNetBlock('Residual_DownSample', hidden_dims[0], hidden_dims[1]),
            Encoder_ResNetBlock('Residual_Stable', hidden_dims[1], hidden_dims[1]),
            Encoder_ResNetBlock('Residual_DownSample', hidden_dims[1], hidden_dims[2]),
            Encoder_ResNetBlock('Residual_Stable', hidden_dims[2], hidden_dims[2])
        )
        self.fc_mu = nn.Linear(flattened_channels, latent_dims)
        self.fc_var = nn.Linear(flattened_channels, latent_dims)

    def forward(self, x):
        # input x.shape = [B,C,H,W] = [Batch_size, Channel, Height, Width]
        x = self.InitialLayer(x)
        x = self.maxpool(x)
        x = self.Encoder_ResidualBlocks(x)

        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]


class Decoder_ResNetBlock(nn.Module):
    """The basic ResNet block architecture for Decoder """
    def __init__( self, flag: str, h_dim_in: int, h_dim_out: int) -> None:
        super().__init__()
        self.flag = flag
        self.relu = nn.ReLU(inplace=True)

        if self.flag == 'Residual_UpSample':  # reduce hidden dimension & up-sample image size
            self.ConvTransposeLayer = nn.Sequential(
                nn.ConvTranspose2d(h_dim_in, h_dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(h_dim_out),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(h_dim_out, h_dim_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(h_dim_out),
            )
            self.ConvTransposeLayer_I_transform = nn.Sequential(
                nn.ConvTranspose2d(h_dim_in, h_dim_out, kernel_size=1, stride=2, padding=0, bias=False, output_padding=1), # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.),
                nn.BatchNorm2d(h_dim_out),
            )

        elif self.flag == 'Residual_Stable':  # Stable signal (same # of channels & same image size.)
            self.ConvTransposeLayer = nn.Sequential(
                nn.ConvTranspose2d(h_dim_in, h_dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(h_dim_out),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(h_dim_out, h_dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(h_dim_out),
            )
        else:
            print("Error: Unknown Decoder_ResNetBlock flag")

    def forward(self, x):
        if self.flag == 'Residual_UpSample':
            identity = self.ConvTransposeLayer_I_transform(x)
        elif self.flag == 'Residual_Stable':
            identity = x
        else:
            print("Error: Unknown Decoder_ResNetBlock flag")
        out = self.ConvTransposeLayer(x)
        out += identity
        x = self.relu(out)
        return x


class ConvDecoder(nn.Module):
    """
    Goal: A convolutional Decoder of Auto-Encoder, based on Residual Net structure.
    """
    def __init__(self, hidden_dims, flattened_channels, latent_dims, image_shape):
        super(ConvDecoder, self).__init__()
        self.image_shape = image_shape
        self.hidden_dims = hidden_dims
        self.last_channels = hidden_dims[-1]
        self.relu = nn.ReLU(inplace=True)

        # ..... reverse the distribution into CNN structure ......
        self.decoder_input = nn.Linear(latent_dims, flattened_channels)

        self.Decoder_ResidualBlocks = nn.Sequential(
            Decoder_ResNetBlock('Residual_Stable', hidden_dims[2], hidden_dims[2]),
            Decoder_ResNetBlock('Residual_UpSample', hidden_dims[2], hidden_dims[1]),
            Decoder_ResNetBlock('Residual_Stable', hidden_dims[1], hidden_dims[1]),
            Decoder_ResNetBlock('Residual_UpSample', hidden_dims[1], hidden_dims[0]),
            Decoder_ResNetBlock('Residual_Stable', hidden_dims[0], hidden_dims[0])
        )

        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')

        self.FinishLayer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], cfg.raw_img_channel, kernel_size=7, stride=2, padding=3,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(cfg.raw_img_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.last_channels, int(self.image_shape[1] / (2 ** cfg.img_down_size_time)),
                   int(self.image_shape[2] / (2 ** cfg.img_down_size_time)) )
        x = self.Decoder_ResidualBlocks(x)
        x = self.unpool(x)
        x = self.FinishLayer(x)
        return x


class VariationalAutoEncoder(nn.Module):
    """
    Goal: The Auto-Encoder, which connect Encoder & Decoder and output encoder embedding.
    """
    def __init__(self, latent_dims, hidden_dims, image_shape):
        super().__init__()
        self.last_channels = hidden_dims[-1]
        # self.img_down_size_time = 2 + len(hidden_dims) - 1  # initial downsize twice (2*2), and
        # downsize 2x in each change of hidden dimension.
        flattened_channels = int(
            self.last_channels * \
            (image_shape[1] / (2 ** cfg.img_down_size_time)) * \
            (image_shape[2] / (2 ** cfg.img_down_size_time)) )

        self.encoder = ConvEncoder(hidden_dims, flattened_channels, latent_dims)
        self.decoder = ConvDecoder(hidden_dims, flattened_channels, latent_dims, image_shape)

    def reparameterize(self, mu, log_var):
        """
        Re-parameterization: Sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), x, mu, log_var, z]

    def forward_encoder(self, x):
        mu, log_var = self.encoder(x)
        return mu, log_var

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes VAE loss function (reconstruction loss + KL divergence)
        """
        recons_loss = nn.functional.binary_cross_entropy(recons, input, reduction="none").sum(dim=[1, 2, 3])
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        loss = (recons_loss + kld_loss).mean(dim=0)
        return loss

    def sample(self, num_samples, device):
        """
        Samples from the latent space and return the corresponding image space map.
        """
        z = torch.randn(num_samples, cfg.latent_dims)
        z = z.to(device)
        samples = self.decoder(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]

    def interpolate(self, starting_inputs, ending_inputs, device, granularity=10):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.
        """
        mu, log_var = self.encoder(starting_inputs.to(device))
        starting_z = self.reparameterize(mu, log_var)  # [B, latent_dims], B=1 here

        mu, log_var = self.encoder(ending_inputs.to(device))
        ending_z = self.reparameterize(mu, log_var)  # [B, latent_dims]

        t = torch.linspace(0, 1, granularity).to(device)  # [granularity]

        intep_line = (
                torch.kron(starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)) +
                torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        )  # [granularity, latent_dims]: generate (granularity) number of latent vectors based interpolation.

        decoded_line = self.decoder(intep_line).reshape(
            (
                starting_inputs.shape[0],
                t.shape[0]
            )
            + (starting_inputs.shape[1:])
        )  # [granularity, input_img_dims]: torch.Size([10, 1, 48, 48]) --> Size([1, 10, 1, 48, 48])
        return decoded_line
