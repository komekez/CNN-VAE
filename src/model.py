import torch
import torch.nn as nn
import torch.nn.functional as F


kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling


class convVAE(nn.Module):
    def __init__(self) -> None:
        super(convVAE,self).__init__()

        #Encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, 
            out_channels=init_channels, 
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, 
            out_channels=init_channels*2, 
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, 
            out_channels=init_channels*4, 
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, 
            out_channels=64, 
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )

        #Fully Connected Layer for learning Representation
        self.fc1 = nn.Linear(64,128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        #Decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels*8,
            kernel_size = kernel_size,
            stride=1,
            padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8,
            out_channels=init_channels*4,
            kernel_size = kernel_size,
            stride=1,
            padding=0
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4,
            out_channels=init_channels*2,
            kernel_size = kernel_size,
            stride=1,
            padding=0
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2,
            out_channels=image_channels,
            kernel_size = kernel_size,
            stride=1,
            padding=0
        ) 

    def reparameterize(self, mu, logVar):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """

        std = torch.exp(0.5*logVar)
        eps = torch.randn_like(std)# `randn_like` as we need the same size
        sample = mu + (eps*std)

        return sample

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x,1).reshape(batch,-1)
        hidden = self.fc1(x)

        #Get mu and logVar
        mu = self.fc_mu(hidden)
        logVar = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, logVar)
        z = self.fc2(z)

        z = z.view(-1, 64, 1, 1)

        #Decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))

        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, logVar