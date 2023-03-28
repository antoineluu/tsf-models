import torch


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def acquire_dataloader(X, y, batch_size: int = 32, shuffle: bool = True):
    return torch.utils.data.DataLoader(TorchDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def acquire_test_tensor(X):
    return torch.from_numpy(X).float()
