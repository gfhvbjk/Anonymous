'''
Helper functions to prepare data for learning tasks
'''
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class PrepareData(Dataset):
    '''
    Create torch dataset
    '''
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def read_data_rl(data_dir):
    '''
    Read in data for contextual bandits
    - transform context and label to one-hot encoded vectors
    '''
    df = pd.read_csv(data_dir, sep=',', header=None, error_bad_lines=False, warn_bad_lines=True, low_memory=False)
    df.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
         'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
         'stalk-surf-above-ring','stalk-surf-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-color','population','habitat']
    X = pd.DataFrame(df, columns=df.columns[1:len(df.columns)], index=df.index)
    Y = df['class']

    # transform to one-hot encoding
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(Y)
    Y_ = label_encoder.transform(Y)
    X_ = X.copy()
    for feature in X.columns:
        label_encoder.fit(X[feature])
        X_[feature] = label_encoder.transform(X[feature])

    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(X_)
    X_ = oh_encoder.transform(X_).toarray()

    return X_, Y_

def create_data_reg(train_size, gap=False):
    '''
    Generate synth data for regression task
    gap = True creates artificial data clusters with gap.
    '''
    np.random.seed(0)
    xs = np.random.uniform(low=0., high=0.6, size=train_size)
    if gap:
        xs[xs>0.3] += 0.4
    eps = np.random.normal(loc=0., scale=0.02, size=[train_size])

    ys = xs + 0.3 * np.sin(2*np.pi * (xs + eps)) + 0.3 * np.sin(4*np.pi * (xs + eps)) + eps

    xs = torch.from_numpy(xs).reshape(-1,1).float()
    ys = torch.from_numpy(ys).reshape(-1,1).float()

    return xs, ys


def create_data_class(train, batch_size, shuffle):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))  # CIFAR-10 的均值和标准差
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    ds = DataLoader(
        datasets.CIFAR10(
            root='./cifar10',
            train=train,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

    return ds