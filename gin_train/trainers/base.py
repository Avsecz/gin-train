import abc
# import os
# import numpy as np
# import pandas as pd
# from gin_train.utils import write_json, prefix_dict
# from tqdm import tqdm
# from collections import OrderedDict
# from kipoi.data_utils import numpy_collate_concat
# from kipoi.external.flatten_json import flatten
# import gin
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Trainer:
    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=(), save=True, **kwargs):
        """
        Evaluate the model on the validation set

        :param metric: a list or a dictionary of metrics
        :param batch_size:
        :param num_workers:
        :param eval_train: if True, also compute the evaluation metrics on the training set
        :param eval_skip:
        :param save: save the json file to the output directory
        :param kwargs:
        :return:
        """
        pass

