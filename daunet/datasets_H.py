from torch.utils.data import Dataset


class matDataset(Dataset):
    def __init__(self, X, Y):
        self.data = {}
        for v, x in enumerate(X):
            self.data[v] = x
        self.Y = Y

    def __getitem__(self, index):
        data = []
        for x in self.data.values():
            data.append(x[index])
        y = self.Y[index]
        return data, y, index

    def __len__(self):
        return len(self.Y)