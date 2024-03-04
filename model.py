from torch import nn
import torch
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 
                               kernel_size=(3,4), 
                               stride=(1,2), 
                               padding=(1,0),),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 
                               kernel_size=3, 
                               stride=(1,2), 
                               padding=(1,0),
                               output_padding=(0,0)),
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(in_features=input_dim, out_features=h_dim)
        self.hid_2mu = nn.Linear(in_features=h_dim,out_features=z_dim)
        self.hid_2sigma = nn.Linear(in_features=h_dim, out_features=z_dim)
        # decoder
        self.z_2hid = nn.Linear(in_features=z_dim, out_features=h_dim)
        self.hid_2img = nn.Linear(in_features=h_dim, out_features=input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h = self.img_2hid(x)
        h = self.relu(h)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decoder(self, z):
        h = self.z_2hid(z)
        h = self.relu(h)
        img = self.hid_2img(h)
        return self.sigmoid(img)
        

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.rand_like(sigma)
        z = mu + sigma * eps
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma
