# gin-train

[![CircleCI](https://circleci.com/gh/Avsecz/gin-train.svg?style=svg&circle-token=b2623a0886aaf8f679e8c2846d162d6bcd5c0c99)](https://circleci.com/gh/Avsecz/gin-train)

This is a simple wrapper around gin-config (https://github.com/google/gin-config/) with connection to [Weights&Biases](https://www.wandb.com/), [comet.ml](https://comet.ml) and S3 to train Keras models and keep track of them. Please see the documentation for [gin-config](https://github.com/google/gin-config) for more information on how to use gin.


## Installation

```bash
pip install gin-train
```

## Examples

- [examples/mnist](examples/mnist) - train a model on the mnist dataset

## Usage

```bash
$ gt --help
usage: gt [-h] [--gin-bindings GIN_BINDINGS] [--gpu GPU]
          [--framework FRAMEWORK] [-c COMETML_PROJECT] [-w WANDB_PROJECT]
          [--remote-dir REMOTE_DIR] [--run-id RUN_ID] [-n NOTE_PARAMS]
          [--force-overwrite]
          gin-files output-dir

Train a model using gin-config

    Args:
      gin_file: comma separated list of gin files
      output_dir: where to store the results. Note: a subdirectory `run_id`
        will be created in `output_dir`.
      gin_bindings: comma separated list of additional gin-bindings to use
      gpu: which gpu to use. Example: gpu=1
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
    

positional arguments:
  gin-files             -
  output-dir            -

optional arguments:
  -h, --help            show this help message and exit
  --gin-bindings GIN_BINDINGS
                        ''
  --gpu GPU             0
  --framework FRAMEWORK
                        'tf'
  -c COMETML_PROJECT, --cometml-project COMETML_PROJECT
                        ''
  -w WANDB_PROJECT, --wandb-project WANDB_PROJECT
                        ''
  --remote-dir REMOTE_DIR
                        ''
  --run-id RUN_ID       -
  -n NOTE_PARAMS, --note-params NOTE_PARAMS
                        ''
  --force-overwrite     False
```

- `gin_file` can be a single or multiple gin files. That allows you to re-use for example the problem definition parts of the
gin config and the model definition part of gin-config.


### Example


```bash
gt problem.gin,model.gin default/ --gpu=1 -c Avsecz/basepair-chipseq-cls -f
```

where the gin files are the following:

- [problem.gin](examples/mnist/problem.gin)
- [model.gin](examples/mnist/model.gin)
