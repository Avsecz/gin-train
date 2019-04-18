import os
import abc

from .base import Trainer

import gin

# import sklearn
from sklearn.pipeline import Pipeline as SklearnModel

import pickle

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class SklearnTrainer(Trainer, metaclass=abc.ABCMeta):
    """Simple Scikit model trainer
    """
    model: SklearnModel

    output_dir: str
    ckp_file: str
    history_path: str
    evaluation_path: str

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None,
                 wandb_run=None):

        logger.info(f"checking model: {type(model)}")
        if not isinstance(model, SklearnModel):
            raise ValueError(f"model is not a {SklearnModel.__module__}.{SklearnModel.__name__}")
        logger.info("model checking passed")
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run
        self.ckp_file = None

        # setup the output directory
        self.set_output_dir(output_dir)

    def set_output_dir(self, output_dir: str):
        """
        Set output folder structure for the model.

        :param output_dir: output directory name
        :return:
        """

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.ckp_file is not None and os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")

        self.ckp_file = os.path.join(self.output_dir, "model.h5")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self, num_workers=8, **kwargs):
        train_data = self.train_dataset.load_all(num_workers=num_workers)
        self.model.fit(train_data["inputs"], train_data["targets"])
        self.save()

    def save(self, **kwargs):
        logger.info(f"saving model to {self.ckp_file}")
        with open(self.ckp_file, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, inputs, **kwargs):
        return self.model.predict(inputs)

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=(), save=True, **kwargs):
        pass
