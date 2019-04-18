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

    @abc.abstractmethod
    def predict(self, inputs):
        """Generic method for predicting using a given model.
           This method has to be implemented in subclass.
        """
        pass

# class Trainer(object):
#     """Generic trainer object. This class has two major components:
#             (i) train model
#             (ii) evalute model (this implies also predicting using the model)
# 
# 
#     Attributes:
#         model: compiled sklearn.pipeline.Pipeline
#         train (:obj:RepDataset): training dataset
#                                 The object should have the following properties:
#                                 obj.inputs, obj.targets, obj.metadata, obj.dataset_name
#         valid (:obj:RepDataset): validation dataset
#         output_dir: output directory where to log the training
#         cometml_experiment: if not None, append logs to commetml
#         wandb_run: send evaluation scores to the dashbord
#     """
# 
#     def __init__(self,
#                  model,
#                  train_dataset,
#                  valid_dataset,
#                  output_dir,
#                  cometml_experiment=None,
#                  wandb_run=None):
# 
#         self.model = model
#         self.train_dataset = train_dataset
#         self.valid_dataset = valid_dataset
#         self.cometml_experiment = cometml_experiment
#         self.wandb_run = wandb_run
# 
#         # setup the output directory
#         self.set_output_dir(output_dir)
# 
#     def set_output_dir(self, output_dir):
#         """Set output folder structure for the model.
# 
#         Args:
#             output_dir (str): output directory name
#         """
# 
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.ckp_file = f"{self.output_dir}/model.h5"
#         if os.path.exists(self.ckp_file):
#             raise ValueError(f"model.h5 already exists in {self.output_dir}")
#         self.history_path = f"{self.output_dir}/history.csv"
#         self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"
# 
#     ##########################################################################
#     ###########################    Train   ###################################
#     ##########################################################################
# 
#     def train(self,
#               batch_size=256,
#               epochs=100,
#               early_stop_patience=4,
#               num_workers=8,
#               train_epoch_frac=1.0,
#               valid_epoch_frac=1.0,
#               train_samples_per_epoch=None,
#               validation_samples=None,
#               train_batch_sampler=None,
#               **kwargs):
# 
#         # **kwargs won't be used, they are just included for compatibility with gin_train.
#         """Train the model
# 
#         Args:
#             num_workers: how many workers to use in parallel
#         """
# 
#         # define dataset
#         X_train, y_train = (self.train_dataset.inputs, self.train_dataset.targets)
#         if self.valid_dataset is None:
#             raise ValueError("len(self.valid_dataset) == 0")
# 
#         # check model type
#         self.check_model()
# 
#         # fit model
#         self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, num_workers=num_workers)
# 
#         # save model
#         self.save()
# 
#     @abc.abstractmethod
#     def check_model(self):
#         """Check if the model has the specified data type."""
#         pass
# 
#     @abc.abstractmethod
#     def fit(self, inputs, targets, **kwargs):
#         """Generic method for fitting using a given model.
#            This method has to be implemented in subclass
#         """
#         pass
# 
#     @abc.abstractmethod
#     def save(self):
#         """Generic method for saving the model.
#            This method has to be implemented in subclass.
#         """
#         pass
# 
#     ##########################################################################
#     ########################### Evaluation ###################################
#     ##########################################################################
# 
#     def evaluate(self,
#                  metric,
#                  batch_size=None,
#                  num_workers=8,
#                  eval_train=False,
#                  eval_skip=False,
#                  save=True):
#         """Evaluate the model on the validation set
# 
#         Args:
#             metric: a list or a dictionary of metrics
#             batch_size: None - means full dataset
#             num_workers: number of threads
#             eval_train: if True, also compute the evaluation metrics on the training set
#             save: save the json file to the output directory
#         """
# 
#         # contruct a list of dataset to evaluate
#         if eval_train:
#             eval_datasets = [self.train_dataset, self.valid_dataset]
#         else:
#             eval_datasets = [self.valid_dataset]
# 
#         metric_res = OrderedDict()
#         eval_metric = metric
# 
#         for i, data in enumerate(eval_datasets):
#             lpreds = []
#             llabels = []
# 
#             inputs = data.inputs
#             targets = data.targets
# 
#             lpreds.append(self.predict(inputs))
#             llabels.append(deepcopy(targets))
# 
#             preds = numpy_collate_concat(lpreds)
#             labels = numpy_collate_concat(llabels)
#             del lpreds
#             del llabels
# 
#             metric_res[data.dataset_name] = eval_metric(labels, preds, tissue_specific_metadata=data.metadata)
# 
#         if save:
#             write_json(metric_res, self.evaluation_path, indent=2)
#             logger.info("Saved metrics to {}".format(self.evaluation_path))
# 
#         if self.cometml_experiment is not None:
#             self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")
# 
#         if self.wandb_run is not None:
#             self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))
# 
#         return metric_res
# 
#     @abc.abstractmethod
#     def predict(self, inputs):
#         """Generic method for predicting using a given model.
#            This method has to be implemented in subclass.
#         """
#         pass
