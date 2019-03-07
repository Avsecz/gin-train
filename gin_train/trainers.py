import os
import numpy as np
import pandas as pd
import logging
import sklearn

from gin_train.utils import write_json, prefix_dict
import gin

from tqdm import tqdm
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from kipoi.data_utils import numpy_collate_concat
from kipoi.external.flatten_json import flatten

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
              train_samples_per_epoch=None,
              validation_samples=None,
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

        if train_samples_per_epoch is None:
            train_steps_per_epoch = max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1)
        else:
            train_steps_per_epoch = max(int(train_samples_per_epoch / batch_size), 1)
            
        
        if validation_samples is None:
            # parametrize with valid_epoch_frac
            validation_steps = max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1)
        else:
            validation_steps = max(int(validation_samples / batch_size), 1)

        self.model.fit_generator(train_it,
                                 epochs=epochs,
                                 steps_per_epoch=train_steps_per_epoch,
                                 validation_data=valid_it,
                                 validation_steps=validation_steps,
                                 callbacks=[EarlyStopping(patience=early_stop_patience,
                                                          restore_best_weights=True),
                                            CSVLogger(self.history_path)] + tb + wcp
                                            # ModelCheckpoint(self.ckp_file, save_best_only=True)] 
                                 )
        self.model.save(self.ckp_file)
        # self.model = load_model(self.ckp_file)  # not necessary, EarlyStopping is already restoring the best weights

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

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=False, save=True, **kwargs):
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
          eval_train: if True, also compute the evaluation metrics on the training set
          save: save the json file to the output directory
        """
        if len(kwargs) > 0:
            logger.warn(f"Extra kwargs were provided to trainer.evaluate: {kwargs}")
        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [('train', self.train_dataset)] + self.valid_dataset
        else:
            eval_datasets = self.valid_dataset

        try:
            if len(eval_skip) > 0:
                eval_datasets = [(k,v) for k,v in eval_datasets if k not in eval_skip]
        except:
            logger.warn(f"eval datasets don't contain tuples. Unable to skip them using {eval_skip}")

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
            from copy import deepcopy
            for inputs, targets in tqdm(dataset.batch_train_iter(cycle=False,
                                                                 num_workers=num_workers,
                                                                 batch_size=batch_size),
                                        total=len(dataset) // batch_size
                                        ):
                lpreds.append(self.model.predict_on_batch(inputs))
                llabels.append(deepcopy(targets))
                del inputs
                del targets
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
    

class Trainer(object):
    """Generic trainer object. This class has two major components:
            (i) train model
            (ii) evalute model (this implies also predicting using the model)
    
        
    Attributes:
        model: compiled sklearn.pipeline.Pipeline
        train (:obj:RepDataset): training dataset 
                                The object should have the following properties: 
                                obj.inputs, obj.targets, obj.metadata, obj.dataset_name
        valid (:obj:RepDataset): validation dataset 
        output_dir: output directory where to log the training
        cometml_experiment: if not None, append logs to commetml
        wandb_run: send evaluation scores to the dashbord
    """
    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):
        
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run

        # setup the output directory
        self.set_output_dir(output_dir)


    def set_output_dir(self, output_dir):
        """Set output folder structure for the model.

        Args:
            output_dir (str): output directory name
        """

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"
    
    
      
    ##########################################################################
    ###########################    Train   ###################################
    ##########################################################################
    
    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              **kwargs):

        # **kwargs won't be used, they are just included for compatibility with gin_train.
        """Train the model
        
        Args:
            num_workers: how many workers to use in parallel
        """
        
        # define dataset
        X_train, y_train = (self.train_dataset.inputs, self.train_dataset.targets)
        if self.valid_dataset is None:
            raise ValueError("len(self.valid_dataset) == 0")

        # check model type
        self.check_model()

        # fit model
        self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, num_workers=num_workers)
        
        # save model
        self.save()
        
    
    @abstractmethod
    def check_model(self):
        """Check if the model has the specified data type."""
        pass
    
    
    @abstractmethod
    def fit(self, inputs, targets):
        """Generic method for fitting using a given model.
           This method has to be implemented in subclass
        """
        pass
    
    
    @abstractmethod
    def save(self):
        """Generic method for saving the model.
           This method has to be implemented in subclass.
        """
        pass
    
    
    ##########################################################################
    ########################### Evaluation ###################################
    ##########################################################################
    

    def evaluate(self, 
                 metric, 
                 batch_size = None,
                 num_workers = 8,
                 eval_train=False,
                 eval_skip=False,
                 save=True):
        """Evaluate the model on the validation set

        Args:
            metric: a list or a dictionary of metrics
            batch_size: None - means full dataset
            num_workers: number of threads
            eval_train: if True, also compute the evaluation metrics on the training set
            save: save the json file to the output directory
        """
        
        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [self.train_dataset, self.valid_dataset]
        else:
            eval_datasets = [self.valid_dataset]
        
        metric_res = OrderedDict()
        eval_metric = metric
        
        for i, data in enumerate(eval_datasets):

            lpreds = []
            llabels = []
            
            inputs = data.inputs
            targets = data.targets
            
            lpreds.append(self.predict(inputs))
            llabels.append(deepcopy(targets))

            preds = numpy_collate_concat(lpreds)
            labels = numpy_collate_concat(llabels)
            del lpreds
            del llabels
            
            metric_res[data.dataset_name] = eval_metric(labels, preds, tissue_specific_metadata = data.metadata)
                
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))

        return metric_res
    
    
    @abstractmethod
    def predict(self, inputs):
        """Generic method for predicting using a given model.
           This method has to be implemented in subclass.
        """
        pass


@gin.configurable
class SklearnPipelineTrainer(Trainer):

    """Simple Scikit model trainer
    """

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):

        Trainer.__init__(self,model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment,
                 wandb_run)

    
    def check_model(self):
        if not isinstance(self.model, sklearn.pipeline.Pipeline):
            raise ValueError("model is not a sklearn.pipeline.Pipeline")
    
    
    def fit(self, inputs, targets, epochs=10, batch_size=256, num_workers=8):        
        self.model.fit(inputs, targets)
    
    
    def save(self):        
        import pickle
        with open(self.ckp_file, 'wb') as file:
            pickle.dump(self.model, file)


    def predict(self, inputs):
        return self.model.predict(inputs)
