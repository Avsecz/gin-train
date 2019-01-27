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
usage: gt [-h] [--run-id RUN_ID] [--gin-bindings GIN_BINDINGS] [--gpu GPU]
          [--force-overwrite] [--framework FRAMEWORK]
          [--remote-dir REMOTE_DIR] [-c COMETML_PROJECT] [-w WANDB_PROJECT]
          [-n NOTE_PARAMS]
          gin-files output-dir

Train a model using gin-config

    Args:
      gin_file: comma separated list of gin files
      gin_bindings: comma separated list of additional gin-bindings to use
      run_id: manual run id
      output_dir: where to store the results
      force_overwrite: if True, the output directory will be overwritten
      cometml_project: comet_ml project name. Example: Avsecz/basepair.
        If not specified, cometml will not get used
      wandb_project: wandb project name. If not specified, wandb will not be used
      note_params: take note of additional key=value pairs.
        Example: --note-params note='my custom note',feature_set=this
    

positional arguments:
  gin-files             -
  output-dir            -

optional arguments:
  -h, --help            show this help message and exit
  --run-id RUN_ID       -
  --gin-bindings GIN_BINDINGS
                        ''
  --gpu GPU             0
  --force-overwrite     False
  --framework FRAMEWORK
                        'tf'
  --remote-dir REMOTE_DIR
                        ''
  -c COMETML_PROJECT, --cometml-project COMETML_PROJECT
                        ''
  -w WANDB_PROJECT, --wandb-project WANDB_PROJECT
                        ''
  -n NOTE_PARAMS, --note-params NOTE_PARAMS

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
