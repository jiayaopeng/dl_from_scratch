import torch
import numpy as np


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4]])
    x_data = torch.from_numpy(data)
    x_data

    x_data_1 = torch.ones_like(x_data)
    x_data_2 = torch.rand_like(x_data, dtype=torch.float)

    x_data_1