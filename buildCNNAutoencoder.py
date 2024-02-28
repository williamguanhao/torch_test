import torch
from torch import nn, optim
from matplotlib.pyplot import plot, scatter
# !pip install rdp
# from rdp import rdp

# 1 batch, 1 channel, 10 tenors, 25 features
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
X = torch.randn(32,1,30,25).to(device)
# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
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
model = Autoencoder().to(device)
model.train()
loss_fn = nn.MSELoss()
loss_list=[]
eps=[]
ep = 0
epochs = 6000
LR=1e-4
optimizer = optim.AdamW(model.parameters(), lr=LR)
for epoch in range(epochs):
    # if epoch > 8000:
    #     LR = 1e-5
    # elif epoch > 6000:
    #     LR = 4e-5
    # elif epoch > 4000:
    #     LR = 1e-4
    # if epoch > 2000:
    #     LR = 3e-4
    # else:
    #     LR = 1e-3
    # optimizer = optim.AdamW(model.parameters(), lr=LR)
    y = model(X)
    loss=loss_fn(y, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print(f"Epoch {epoch}: {loss.item()}, using learing rate: {LR}")
        loss_list.append(loss.item())
        ep += 100
        eps.append(ep)
# ep = torch.arange(100,5100,100)
plot(eps,loss_list)

# test shape
X = torch.randn(1,2,30,25)
conv1 = nn.Conv2d(2,16,kernel_size=3,stride=1,padding=(1,1))
c1=conv1(X)
c1.shape
mp1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2))
m1=mp1(c1)
m1.shape
conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=(1,1))
c2=conv2(m1)
c2.shape
mp2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2))
m2=mp2(c2)
m2.shape
y=torch.randn(1,32,30,5)
transcov1 = nn.ConvTranspose2d(32, 16, 
                               kernel_size=3, 
                               stride=(1,2), 
                               padding=(1,0),
                               output_padding=(0,1))

t1 = transcov1(y)
t1.shape
transcov2 = nn.ConvTranspose2d(16, 2, 
                               kernel_size=3, 
                               stride=(1,2), 
                               padding=(1,0),
                               output_padding=(0,0))
t2 = transcov2(t1)
t2.shape
