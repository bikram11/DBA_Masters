import torch
from torch.utils.data import Dataset


class Sequencial_Dataloader(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if index >= self.sequence_length +1:
            index_start = index - self.sequence_length + 1
            x = self.X[index_start:(index+1),:]
        else:
            padding = self.X[0].repeat(self.sequence_length-index-1,1)
            x = self.X[0:(index+1),:]
            x = torch.cat((padding,x),0)

        return x, self.y[index]

    