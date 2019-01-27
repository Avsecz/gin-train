## Train an mnist model

cd into this directory: 

```bash
cd examples/mnist
```

Run:
```bash
gt config.gin output_dir
```

### Example with cometml and s3:

```bash
gt config.gin output_dir --remote-dir=s3://bucket1/test/asd --cometml-project Avsecz/test
```

### Example with wandb

```bash
gt config.gin output_dir --wandb-project Avsecz/test
```

### Example with multiple evaluation datasets

```bash
gt config-multiple-eval-datasets.gin output_dir --wandb-project Avsecz/test
```