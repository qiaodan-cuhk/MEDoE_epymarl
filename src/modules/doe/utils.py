import torch
from torch.utils.data import Dataset


class SimpleListDataset(Dataset):
        
    def __init__(self, x, y):
        # x
        if isinstance(x, torch.Tensor):
            self.x_train = torch.clone(x).detach().float()
        else:
            self.x_train = torch.tensor(x, dtype=torch.float32)
        # y
        if isinstance(y, torch.Tensor):
            self.y_train = torch.clone(y).detach().float()
        else:
            self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
