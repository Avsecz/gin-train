# gin-train

[![CircleCI](https://circleci.com/gh/Avsecz/gin-train.svg?style=svg&circle-token=b2623a0886aaf8f679e8c2846d162d6bcd5c0c99)](https://circleci.com/gh/Avsecz/gin-train)

This is a simple wrapper around gin-config (https://github.com/google/gin-config/) to train Keras models. Please see the documentation for gin-config for more information on how to use gin.


## Installation

`git clone git@github.com:Avsecz/gin-train.git`
`pip install -e gin-train`

## Examples

- [examples/mnist](examples/mnist) - train a model on the mnist dataset

## Usage

```bash
$ gin_train --help
usage: gin_train [-h] [--gin-bindings GIN_BINDINGS] [--gpu GPU]
                 [--force-overwrite] [--framework FRAMEWORK]
                 [--cometml-project COMETML_PROJECT]
                 [--cometml-log COMETML_LOG]
                 gin-files output-dir

Train a model using gin-config
    Args:
      gin_file: comma separated list of gin files
      gin_bindings: comma separated list of additional gin-bindings to use
      output_dir: where to store the results
      force_overwrite: if True, the output directory will be overwritten
      cometml_project: comet_ml project name. Example: Avsecz/basepair.
        If not specified, cometml will not get used
      cometml_log: additional notes for cometml
    

positional arguments:
  gin-files             -
  output-dir            -

optional arguments:
  -h, --help            show this help message and exit
  --gin-bindings GIN_BINDINGS
                        ''
  --gpu GPU             0
  --force-overwrite     False
  --framework FRAMEWORK
                        'tf'
  --cometml-project COMETML_PROJECT
                        ''
  --cometml-log COMETML_LOG
                        ''
```

- `gin_file` can be a single or multiple gin files. That allows you to re-use for example the problem definition parts of the
gin config and the model definition part of gin-config.


### Example


```bash
gin_train problem.gin,model.gin default/ --gpu=1 -c Avsecz/basepair-chipseq-cls -f
```

where the gin files are the following:

- [problem.gin](examples/mnist/problem.gin)
- [model.gin](examples/mnist/model.gin)
