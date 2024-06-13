from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from torch.utils.data import DataLoader, Dataset


class Data(Dataset):
    def __init__(self, X_input, y_input):
        self.X = X_input
        self.y = y_input
        self.len = X_input.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == '__main__':
    X_, y_ = make_circles(n_samples=10000, noise=0.05, random_state=26)

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=.33, random_state=26)
    train_data = Data(X_train, y_train)
    test_data = Data(X_test, y_test)

    batch_size = 64
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for batch, (X, y) in enumerate(train):
        print(f'batch number {batch}')
        print(X.shape)
        break

