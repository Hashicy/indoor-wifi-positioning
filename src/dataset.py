"""PyTorch Dataset and DataLoader for the Wi-Fi fingerprints."""

import torch
from torch.utils.data import DataLoader, Dataset


class WifiDataset(Dataset):
    """Holds one data set (train, val or test) from prepare_data()."""

    def __init__(self, d):
        self.X = torch.tensor(d["X"], dtype=torch.float32)
        self.building = torch.tensor(d["building"], dtype=torch.long)
        self.floor = torch.tensor(d["floor"], dtype=torch.long)
        self.coords = torch.tensor(d["coords"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.building[i], self.floor[i], self.coords[i]


def make_loader(d, batch_size=128, shuffle=False):
    return DataLoader(WifiDataset(d), batch_size=batch_size, shuffle=shuffle)
