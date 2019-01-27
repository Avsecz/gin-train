"""TODO - mnist datasets
"""
import keras.layers as kl
from keras.models import Sequential
import numpy as np
from kipoi.data import Dataset
import gin.tf
# TODO _ setup a random dataloader


@gin.configurable
class RandomDataset(Dataset):
    def __init__(self, n=1000):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"inputs": np.random.randn(64),
                "targets": np.random.randint(0, 2)  # binary class {0,1}
                }


@gin.configurable
def train_test_data(dataset_cls, n=1000):
    return dataset_cls(), RandomDataset(int(n * 0.1))


@gin.configurable
def multiple_train_test_data(dataset_cls, n=1000):
    """Example of multiple train and evaluation datasets.

    Second one has a special evaluation metric
    """
    from gin_train.metrics import MetricsDict, accuracy
    return dataset_cls(), [("valid1", RandomDataset(int(n * 0.1))),
                           ("valid2", RandomDataset(int(n * 0.1)), MetricsDict({"accuracy": accuracy}))]


@gin.configurable
def mlp_model(n_hidden=10):
    """Function compiling the Keras model
    """
    m = Sequential([
        kl.Dense(n_hidden, activation='relu', input_shape=(64,)),
        kl.Dense(1)
    ])
    m.compile("adam", "binary_crossentropy", ['accuracy'])
    return m
