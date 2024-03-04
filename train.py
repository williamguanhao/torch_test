import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import data_loader, engine, model
from torchvision import transforms
from torchinfo import summary
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Hyperparameters setup
NUM_EPOCHS = 1000
BATCH_SIZE =16
LEARNING_RATE = 3e-4
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# Directory setup
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Device setup
# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")

# Create transform
data_transform = transforms.Compose([
    
])

# Load data
# arbitrary data with same data shape, size 10 * 30 Batch * 30 tenors, 25 features. 
df = pd.DataFrame(np.random.randn(9000, 25), columns=np.arange(1,26,1))
df['idx']=None
for n in range(300):
   df['idx'].iloc[n*30:n*30+30]= n
risk_dataset = data_loader.CustomDataset(df, normalize=True)
train_dataloader = data_loader.DataLoader(dataset=risk_dataset, batch_size=4,shuffle=True)
test_dataloader = data_loader.DataLoader(dataset=risk_dataset, batch_size=4,shuffle=True)

# Build model
ae = model.CNNAutoencoder()
summary(model=ae, 
        input_size=(16, 1, 30, 25), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
# Loss function and optimizer setup
loss_fn = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=LEARNING_RATE)

# Actual train
engine.train(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    model=ae,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device
)

# utils.save_model(model=vgg_mod,
#                  target_dir='models',
#                  model_name='EffNet_model.pth')