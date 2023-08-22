"""
Taken from neuraloperator/neuralop/datasets/tensor_dataset.py
Origianlly by: Nikola Kovachki (https://github.com/kovachki)
Modified by Keith Poletti 06/14/2023
Only adds metadata to the dataset
"""
from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    """
    A PyTorch Dataset class that represents a dataset consisting of input and output
    tensors, along with optional metadata for each tensor. This class inherits from the
    PyTorch Dataset class.

    Args:
    - Dataset (torch.utils.data.Dataset): PyTorch Dataset class that this class inherits
    """

    def __init__(
        self,
        tensor_x,
        tensor_y,
        meta_x=None,
        meta_y=None,
        transform_x=None,
        transform_y=None,
    ):
        """
        Initializes a TensorDataset object.

        Args:
        - x (torch.Tensor): input tensor
        - y (torch.Tensor): output tensor
        - meta_x (list, optional): metadata for input tensor
        - meta_y (list, optional): metadata for output tensor
        - transform_x (callable, optional): function to transform input tensor
        - transform_y (callable, optional): function to transform output tensor
        """
        assert tensor_x.size(0) == tensor_y.size(0), "Size mismatch between tensors"
        self.x = tensor_x
        self.y = tensor_y
        self.meta_x = meta_x
        self.meta_y = meta_y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        """
        Returns a dictionary containing the input tensor, output tensor, and their
        respective metadata.

        Args:
        - index (int): index of the data point to retrieve

        Returns:
        - dict: dictionary containing the input tensor, output tensor, and their
        respective metadata
        """
        tensor_x = self.x[index]
        tensor_y = self.y[index]
        meta_x = self.meta_x[index]
        meta_y = self.meta_y[index]
        if self.transform_x is not None:
            tensor_x = self.transform_x(tensor_x)

        if self.transform_y is not None:
            tensor_x = self.transform_y(tensor_x)

        return {"x": tensor_x, "y": tensor_y, "meta_x": meta_x, "meta_y": meta_y}

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
        - int: number of data points in the dataset
        """
        return self.x.size(0)
