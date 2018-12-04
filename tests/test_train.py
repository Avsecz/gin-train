"""test run gin
"""
from kipoi.data import Dataset
import numpy as np
import os
from gin_train.cli.gin_train import gin_train
import gin


@gin.configurable
class Dummy(Dataset):
    def __init__(self, n,
                 incl_chromosomes=None,
                 excl_chromosomes=None):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"inputs": np.array([idx, idx + 1]),
                "targets": idx // 2
                }


@gin.configurable
def dummy_model(n_hidden, lr=0.04):
    import keras.layers as kl
    from keras.models import Model
    inp = kl.Input((2,))
    x = kl.Dense(n_hidden)(inp)
    x = kl.Dense(1)(x)
    model = Model([inp], x)
    model.compile('Adam', loss="mse")
    return model


@gin.configurable
def train_valid_dataset(dataset_cls):
    return dataset_cls(), dataset_cls()


def test_gin_train(tmpdir):
    gin_train("tests/data/example.gin", str(tmpdir), force_overwrite=True)

    # produced files
    # assert os.path.exists(os.path.join(str(tmpdir), "log/stdout.log"))
    assert os.path.exists(os.path.join(str(tmpdir), "config.gin"))
    assert os.path.exists(os.path.join(str(tmpdir), "model.h5"))
    assert os.path.exists(os.path.join(str(tmpdir), "history.csv"))
