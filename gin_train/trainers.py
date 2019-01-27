import os
import numpy as np
import pandas as pd
from gin_train.utils import write_json, prefix_dict
from tqdm import tqdm
from collections import OrderedDict
from kipoi.data_utils import numpy_collate_concat
from kipoi.external.flatten_json import flatten
import gin
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class KerasTrainer:
    """Simple Keras model trainer
    """

    def __init__(self, model, train_dataset, valid_dataset, output_dir,
                 cometml_experiment=None, wandb_run=None):
        """
        Args:
          model: compiled keras.Model
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run

        if not isinstance(self.valid_dataset, list):
            # package the validation dataset into a list of validation datasets
            self.valid_dataset = [('valid', self.valid_dataset)]

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_batch_sampler=None,
              tensorboard=True):
        """Train the model
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
        """
        from keras.callbacks import EarlyStopping, History, CSVLogger, ModelCheckpoint, TensorBoard
        from keras.models import load_model

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        next(train_it)
        valid_dataset = self.valid_dataset[0][1]  # take the first one
        valid_it = valid_dataset.batch_train_iter(batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        next(valid_it)

        if tensorboard:
            tb = [TensorBoard(log_dir=self.output_dir)]
        else:
            tb = []

        if self.wandb_run is not None:
            from wandb.keras import WandbCallback
            wcp = [WandbCallback(save_model=False)]  # we save the model using ModelCheckpoint
        else:
            wcp = []

        # train the model
        if len(valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        self.model.fit_generator(train_it,
                                 epochs=epochs,
                                 steps_per_epoch=max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1),
                                 validation_data=valid_it,
                                 validation_steps=max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1),
                                 callbacks=[EarlyStopping(patience=early_stop_patience),
                                            CSVLogger(self.history_path),
                                            ModelCheckpoint(self.ckp_file, save_best_only=True)] + tb + wcp
                                 )
        self.model = load_model(self.ckp_file)

        # log metrics from the best epoch
        dfh = pd.read_csv(self.history_path)
        m = dict(dfh.iloc[dfh.val_loss.idxmin()])
        if self.cometml_experiment is not None:
            self.cometml_experiment.log_multiple_metrics(m, prefix="best-epoch/")
        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(m, prefix="best-epoch/"), separator='/'))

    #     def load_best(self):
    #         """Load the best model from the Checkpoint file
    #         """
    #         self.model = load_model(self.ckp_file)

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, save=True):
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
          eval_train: if True, also compute the evaluation metrics on the training set
          save: save the json file to the output directory
        """
        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [('train', self.train_dataset)] + self.valid_dataset
        else:
            eval_datasets = self.valid_dataset

        metric_res = OrderedDict()
        for d in eval_datasets:
            if len(d) == 2:
                dataset_name, dataset = d
                eval_metric = metric  # use the default eval metric
            elif len(d) == 3:
                # specialized evaluation metric was passed
                dataset_name, dataset, eval_metric = d
            else:
                # TODO - this should be made more explicit with classes
                raise ValueError("Valid dataset needs to be a list of tuples of 2 or 3 elements"
                                 "(name, dataset) or (name, dataset, metric)")
            logger.info(f"Evaluating dataset: {dataset_name}")
            lpreds = []
            llabels = []
            for inputs, targets in tqdm(dataset.batch_train_iter(cycle=False,
                                                                 num_workers=num_workers,
                                                                 batch_size=batch_size),
                                        total=len(dataset) // batch_size
                                        ):
                lpreds.append(self.model.predict_on_batch(inputs))
                llabels.append(targets)
            preds = numpy_collate_concat(lpreds)
            labels = numpy_collate_concat(llabels)
            del lpreds
            del llabels
            metric_res[dataset_name] = eval_metric(labels, preds)

        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))

        return metric_res
