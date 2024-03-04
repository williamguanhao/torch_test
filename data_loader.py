import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# arbitrary data with same data shape, size 10 * 30 Batch * 30 tenors, 25 features. 
df = pd.DataFrame(np.random.randn(9000, 25), columns=np.arange(1,26,1))
df['idx']=None
for n in range(300):
#    df.loc[]
   df['idx'].iloc[n*30:n*30+30]= n
   

class CustomDataset(Dataset):
  def __init__(self, data, normalize:bool=True):
    self.data = data
    self.normalize = normalize
    '''
    input: dataframe
    normalize across dim 2 i.e. features
    '''
  def __len__(self):
    return int(len(self.data)/30)

  def __getitem__(self, idx):
        # accessing data at particular indexes
    sample = self.data[self.data['idx']==idx].iloc[:,0:25].values
    sample_tensor = torch.tensor(sample).unsqueeze(0).float()
    if self.normalize:
        magnitude = sample_tensor.norm(p=2, dim=2, keepdim=True)
        sample_tensor = normalize(sample_tensor, dim=2, p=2) 
    else:
        magnitude=None
    return sample_tensor, magnitude, idx

risk_dataset = CustomDataset(df, normalize=True)
train_dataloader = DataLoader(dataset=risk_dataset, batch_size=4,shuffle=True)
train_data, mag, idx = next(iter(train_dataloader))
