from torch.utils.data.dataset import Dataset

"""
Taken from neuraloperator/neuralop/datasets/tensor_dataset.py
Origianlly by: Nikola Kovachki (https://github.com/kovachki)
Modified by Keith Poletti 06/14/2023
Only adds metadata to the dataset
"""


class TensorDataset(Dataset):
    def __init__(
        self, x, y, meta_x=None, meta_y=None, transform_x=None, transform_y=None
    ):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.meta_x = meta_x
        self.meta_y = meta_y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        meta_x = self.meta_x[index]
        meta_y = self.meta_y[index]
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            x = self.transform_y(x)

        return {"x": x, "y": y, "meta_x": meta_x, "meta_y": meta_y}

    def __len__(self):
        return self.x.size(0)
