"""Train models using gin configuration
"""
import gin
import json
import sys
import os
import yaml
from uuid import uuid4
from fs.osfs import OSFS
from tqdm import tqdm
from gin_train.remote import upload_dir
from gin_train.config import create_tf_session
from gin_train.utils import write_json, Logger, NumpyAwareJSONEncoder, prefix_dict
from comet_ml import Experiment

try:
    import wandb
except ImportError:
    wandb = None
# import all modules registering any gin configurables

# configurables import
from gin_train.trainers import KerasTrainer
from gin_train import metrics
from gin_train import trainers
from gin_train import samplers


import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def log_gin_config(output_dir, cometml_experiment=None, wandb_run=None):
    gin_config_str = gin.operative_config_str()

    print("Used config: " + "-" * 40)
    print(gin_config_str)
    print("-" * 52)
    with open(os.path.join(output_dir, "config.gin"), "w") as f:
        f.write(gin_config_str)
    # parse the gin config string to dictionary
    gin_config_str = "\n".join([x for x in gin_config_str.split("\n")
                                if not x.startswith("import")])
    gin_config_dict = yaml.load(gin_config_str
                                .replace("@", "")
                                .replace(" = %", ": ")
                                .replace(" = ", ": "))
    write_json(gin_config_dict,
               os.path.join(output_dir, "config.gin.json"),
               sort_keys=True,
               indent=2)

    if cometml_experiment is not None:
        # Skip any rows starting with import
        cometml_experiment.log_multiple_params(gin_config_dict)

    if wandb_run is not None:
        # This allows to display the metric on the dashboard
        wandb_run.config.update({k.replace(".", "/"): v for k, v in gin_config_dict.items()})


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
          eval_metric=None,
          eval_train=False,
          trainer_cls=KerasTrainer,
          # shared
          batch_size=256,
          num_workers=8,
          # train-specific
          epochs=100,
          early_stop_patience=4,
          train_epoch_frac=1.0,
          valid_epoch_frac=1.0,
          train_samples_per_epoch=None,
          validation_samples=None,
          train_batch_sampler=None,
          stratified_sampler_p=None,
          tensorboard=True,
          remote_dir='',
          cometml_experiment=None,
          wandb_run=None
          ):
    """Main entry point to configure in the gin config
    Args:
      model: compiled keras model
      data: tuple of (train, valid) Datasets
      eval_train: if True, also compute the evaluation metrics for the final model
        on the training set
    """
    # from this point on, no configurable should be added. Save the gin config
    log_gin_config(output_dir, cometml_experiment, wandb_run)

    train_dataset, valid_dataset = data[0], data[1]

    # make sure the validation dataset names are unique
    if isinstance(valid_dataset, list):
        dataset_names = []
        for d in valid_dataset:
            dataset_name = d[0]
            if dataset_name in dataset_names:
                raise ValueError("The dataset names are not unique")
            dataset_names.append(dataset_name)

    if stratified_sampler_p is not None and train_batch_sampler is not None:
        raise ValueError("stratified_sampler_p and train_batch_sampler are mutually exclusive."
                         " Please specify only one of them.")

    if stratified_sampler_p is not None and train_batch_sampler is None:
        # HACK - there is no guarantee that train_dataset.get_targets() will exist
        # Maybe we have to introduce a ClassificationDataset instead which will
        # always implement get_targets()
        logger.info(f"Using stratified samplers with p: {stratified_sampler_p}")
        train_batch_sampler = samplers.StratifiedRandomBatchSampler(train_dataset.get_targets().max(axis=1),
                                                                    batch_size=batch_size,
                                                                    p_vec=stratified_sampler_p,
                                                                    verbose=True)

    tr = trainer_cls(model, train_dataset, valid_dataset, output_dir, cometml_experiment, wandb_run)
    tr.train(batch_size=batch_size, 
             epochs=epochs, 
             early_stop_patience=early_stop_patience,
             num_workers=num_workers, 
             train_epoch_frac=train_epoch_frac, 
             valid_epoch_frac=valid_epoch_frac,
             train_samples_per_epoch=train_samples_per_epoch,
             validation_samples=validation_samples,
             train_batch_sampler=train_batch_sampler, 
             tensorboard=tensorboard)
    final_metrics = tr.evaluate(eval_metric, batch_size=batch_size, num_workers=num_workers,
                                eval_train=eval_train, save=True)
    # pass
    logger.info("Done!")
    print("-" * 40)
    print("Final metrics: ")
    print(json.dumps(final_metrics, cls=NumpyAwareJSONEncoder, indent=2))

    # upload files to a custom remote directory
    if remote_dir:
        logger.info("Uploading files to: {}".format(remote_dir))
        upload_dir(output_dir, remote_dir)

    # upload files to comet.ml
    if cometml_experiment is not None:
        logger.info("Uploading files to comet.ml")
        for f in tqdm(list(OSFS(output_dir).walk.files())):
            # [1:] removes trailing slash
            cometml_experiment.log_asset(file_path=os.path.join(output_dir, f[1:]),
                                         file_name=f[1:])
    return final_metrics


def gin_train(gin_files,
              output_dir,
              gin_bindings='',
              gpu=0,
              memfrac=0.45,
              framework='tf',
              cometml_project="",
              wandb_project="",
              remote_dir="",
              run_id=None,
              note_params="",
              force_overwrite=False):
    """Train a model using gin-config

    Args:
      gin_file: comma separated list of gin files
      output_dir: where to store the results. Note: a subdirectory `run_id`
        will be created in `output_dir`.
      gin_bindings: comma separated list of additional gin-bindings to use
      gpu: which gpu to use. Example: gpu=1
      memfrac: what fraction of the GPU's memory to use
      framework: which framework to use. Available: tf
      cometml_project: comet_ml project name. Example: Avsecz/basepair.
        If not specified, cometml will not get used
      wandb_project: wandb `<entity>/<project>` name. Example: Avsecz/test.
        If not specified, wandb will not be used
      remote_dir: additional path to the remote directory. Can be an s3 path.
        Example: `s3://mybucket/model1/exp1`
      run_id: manual run id. If not specified, it will be either randomly
        generated or re-used from wandb or comet.ml.
      note_params: take note of additional key=value pairs.
        Example: --note-params note='my custom note',feature_set=this
      force_overwrite: if True, the output directory will be overwritten
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

    if wandb_project:
        assert "/" in wandb_project
        entity, project = wandb_project.split("/")
        if wandb is None:
            logger.warn("wandb not installed. Not using it")
            wandb_run = None
        else:
            wandb._set_stage_dir("./")  # Don't prepend wandb to output file
            if run_id is not None:
                wandb.init(project=project,
                           dir=output_dir,
                           entity=entity,
                           resume=run_id)
            else:
                # automatically set the output
                wandb.init(project=project,
                           entity=entity,
                           dir=output_dir)
            wandb_run = wandb.run
            logger.info("Using wandb")
            print(wandb_run)
    else:
        wandb_run = None

    # update the output directory
    if run_id is None:
        if wandb_run is not None:
            run_id = os.path.basename(wandb_run.dir)
        elif cometml_experiment is not None:
            run_id = cometml_experiment.id
        else:
            # random run_id
            run_id = str(uuid4())
    output_dir = os.path.join(output_dir, run_id)
    if remote_dir:
        remote_dir = os.path.join(remote_dir, run_id)
    if wandb_run is not None:
        # make sure the output directory is the same
        # wandb_run._dir = os.path.normpath(output_dir)  # This doesn't work
        # assert os.path.normpath(wandb_run.dir) == os.path.normpath(output_dir)
        # TODO - fix this assertion-> the output directories should be the same
        # in order for snakemake to work correctly
        pass
    # -----------------------------

    if os.path.exists(os.path.join(output_dir, 'config.gin')):
        if force_overwrite:
            logger.info(f"config.gin already exists in the output "
                        "directory {output_dir}. Removing the whole directory.")
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} shouldn't exist!")
    os.makedirs(output_dir, exist_ok=True)  # make the output directory. It shouldn't exist

    # add logging to the file
    add_file_logging(output_dir, logger)

    if framework == 'tf':
        import gin.tf
        if gpu is not None:
            logger.info(f"Using gpu: {gpu}, memory fraction: {memfrac}")
            create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac)

    gin.parse_config_files_and_bindings(gin_files.split(","),
                                        bindings=gin_bindings.split(";"),
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
               indent=2)

    # comet - log environment
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
                   indent=2)

    # wandb - log environment
    if wandb_run is not None:
        write_json({"url": wandb_run.get_url(),
                    "key": wandb_run.id,
                    "project": wandb_run.project,
                    "path": wandb_run.path,
                    "group": wandb_run.group
                    },
                   os.path.join(output_dir, "wandb.json"),
                   sort_keys=True,
                   indent=2)
        # store general configs
        wandb_run.config.update(prefix_dict(dict(gin_files=gin_files,
                                                 gin_bindings=gin_bindings,
                                                 output_dir=output_dir,
                                                 gpu=gpu), prefix='cli/'))
        wandb_run.config.update(note_params_dict)

    if remote_dir:
        import time
        logger.info("Test file upload to: {}".format(remote_dir))
        upload_dir(output_dir, remote_dir)
    return train(output_dir=output_dir, remote_dir=remote_dir, cometml_experiment=cometml_experiment, wandb_run=wandb_run)
