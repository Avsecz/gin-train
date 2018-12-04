## Train an mnist model

cd into this directory: 

```bash
cd examples/mnist
```

Run:
```bash
gt config.gin output_dir --force-overwrite
```

Example with cometml and s3:

```bash
gt config.gin output_dir/a12e --remote-dir=s3://bucket1/test/asd --auto-subdir --cometml-project Avsecz/test
```