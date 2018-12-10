"""Train models using gin configuration
"""
import gin
import json
import sys
import os
import yaml
from uuid import uuid4
from gin_train.remote import upload_dir
from gin_train.config import create_tf_session
from gin_train.utils import write_json, Logger, NumpyAwareJSONEncoder
from comet_ml import Experiment

# import all modules registering any gin configurables

# configurables import
from gin_train.trainers import KerasTrainer
from gin_train import metrics
from gin_train import trainers
from gin_train import samplers


import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def log_gin_config(output_dir, cometml_experiment=None):
    gin_config_str = gin.operative_config_str()

    print("Used config: " + "-" * 40)
    print(gin_config_str)
    print("-" * 52)
    with open(os.path.join(output_dir, "config.gin"), "w") as f:
        f.write(gin_config_str)
    # parse the gin config string to dictionary
    gin_config_str = "\n".join([x for x in gin_config_str.split("\n")
                                if not x.startswith("import")])
    gin_config_dict = yaml.load(gin_config_str.replace(" = @", ": ").replace(" = ", ": "))
    write_json(gin_config_dict,
               os.path.join(output_dir, "config.gin.json"),
               sort_keys=True,
               indent=4)

    if cometml_experiment is not None:
        # Skip any rows starting with import
        cometml_experiment.log_multiple_params(gin_config_dict)


def add_file_logging(output_dir, logger, name='stdout'):
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)
    log = Logger(os.path.join(output_dir, 'log', name + '.log'), 'a+')  # log to the file
    fh = logging.FileHandler(os.path.join(output_dir, 'log', name + '.log'), 'a+')
    fh.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return log


def kv_string2dict(s):
    """Convert a key-value string: k=v,k2=v2,... into a dictionary
    """
    return yaml.load(s.replace(",", "\n").replace("=", ": "))


@gin.configurable
def train(output_dir,
          model=gin.REQUIRED,
          data=gin.REQUIRED,
          eval_metric=gin.REQUIRED,
          trainer_cls=KerasTrainer,
          # shared
          batch_size=256,
          num_workers=8,
          # train-specific
          epochs=100,
          early_stop_patience=4,
          train_epoch_frac=1.0,
          valid_epoch_frac=1.0,
          train_batch_sampler=None,
          stratified_sampler_p=None,
          tensorboard=True,
          remote_dir=None,
          cometml_experiment=None
          ):
    """Main entry point to configure in the gin config
    Args:
      model: compiled keras model
      data: tuple of (train, valid) Datasets
    """
    # from this point on, no configurable should be added. Save the gin config
    log_gin_config(output_dir, cometml_experiment)

    train_dataset, valid_dataset = data
    if stratified_sampler_p is not None and train_batch_sampler is None:
        # HACK - there is no guarantee that train_dataset.get_targets() will exist
        # Maybe we have to introduce a ClassificationDataset instead which will
        # always implement get_targets()
        logger.info(f"Using stratified samplers with p: {stratified_sampler_p}")
        train_batch_sampler = samplers.StratifiedRandomBatchSampler(train_dataset.get_targets().max(axis=1),
                                                                    batch_size=batch_size,
                                                                    p_vec=stratified_sampler_p,
                                                                    verbose=True)
    if stratified_sampler_p is not None and train_batch_sampler is not None:
        raise ValueError("stratified_sampler_p and train_batch_sampler are mutually exclusive."
                         " Please specify only one of them.")

    tr = trainer_cls(model, train_dataset, valid_dataset, output_dir, cometml_experiment)
    tr.train(batch_size, epochs, early_stop_patience,
             num_workers, train_epoch_frac, valid_epoch_frac, train_batch_sampler, tensorboard)
    final_metrics = tr.evaluate(eval_metric, batch_size=batch_size, num_workers=num_workers)
    logger.info("Done!")
    print("-" * 40)
    print("Final metrics: ")
    print(json.dumps(final_metrics, cls=NumpyAwareJSONEncoder, indent=2))
    if remote_dir is not None:
        import time
        time.sleep(1)  # sleep so that hdf5 from Keras finishes writing
        logger.info("Uploading files to: {}".format(remote_dir))
        upload_dir(output_dir, remote_dir)
    return final_metrics


def gin_train(gin_files, output_dir,
              gin_bindings='',
              gpu=0,
              force_overwrite=False,
              framework='tf',
              auto_subdir=False,
              remote_dir="",
              cometml_project="",
              note_params=""):
    """Train a model using gin-config

    Args:
      gin_file: comma separated list of gin files
      gin_bindings: comma separated list of additional gin-bindings to use
      output_dir: where to store the results
      force_overwrite: if True, the output directory will be overwritten
      cometml_project: comet_ml project name. Example: Avsecz/basepair.
        If not specified, cometml will not get used
      note_params: take note of additional key=value pairs.
        Example: --note-params note='my custom note',feature_set=this
    """

    sys.path.append(os.getcwd())
    if cometml_project:
        logger.info("Using comet.ml")
        workspace, project_name = cometml_project.split("/")
        cometml_experiment = Experiment(project_name=project_name, workspace=workspace)
        # TODO - get the experiment id
        # specify output_dir to that directory
    else:
        cometml_experiment = None

    if auto_subdir:
        if cometml_experiment is None:
            # create random uuid
            output_dir = os.path.join(output_dir, str(uuid4()))
            if remote_dir:
                remote_dir = os.path.join(remote_dir, str(uuid4()))
        else:
            # use Comet.ml experiment
            output_dir = os.path.join(output_dir, cometml_experiment.id)
            if remote_dir:
                remote_dir = os.path.join(remote_dir, cometml_experiment.id)

    if os.path.exists(output_dir):
        if force_overwrite:
            logger.info(f"Output directory exists: {output_dir}. Removing it.")
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} shouldn't exist!")
    os.makedirs(output_dir)  # make the output directory. It shouldn't exist

    # add logging to the file
    add_file_logging(output_dir, logger)

    if framework == 'tf':
        import gin.tf
        if gpu is not None:
            logger.info(f"Using gpu: {gpu}")
            create_tf_session(gpu)

    gin.parse_config_files_and_bindings(gin_files.split(","),
                                        bindings=gin_bindings.split(","),
                                        skip_unknown=False)

    # write note_params.json
    if note_params:
        logger.info(f"note_params: {note_params}")
        note_params_dict = kv_string2dict(note_params)
    else:
        note_params_dict = dict()
    write_json(note_params_dict,
               os.path.join(output_dir, "note_params.json"),
               sort_keys=True,
               indent=4)

    if cometml_experiment is not None:
        # log other parameters
        cometml_experiment.log_multiple_params(dict(gin_files=gin_files,
                                                    gin_bindings=gin_bindings,
                                                    output_dir=output_dir,
                                                    gpu=gpu), prefix='cli/')
        cometml_experiment.log_multiple_params(note_params_dict)

        exp_url = f"https://www.comet.ml/{cometml_experiment.workspace}/{cometml_experiment.project_name}/{cometml_experiment.id}"
        logger.info("Comet.ml url: " + exp_url)
        # write the information about comet.ml experiment
        write_json({"url": exp_url,
                    "key": cometml_experiment.id,
                    "project": cometml_experiment.project_name,
                    "workspace": cometml_experiment.workspace},
                   os.path.join(output_dir, "cometml.json"),
                   sort_keys=True,
                   indent=4)

    if remote_dir:
        import time
        logger.info("Test file upload to: {}".format(remote_dir))
        upload_dir(output_dir, remote_dir)
    return train(output_dir=output_dir, remote_dir=remote_dir, cometml_experiment=cometml_experiment)
