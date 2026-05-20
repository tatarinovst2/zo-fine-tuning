# Pipeline

To reproduce the results, run the following commands depending on the task:

## Classification task

### Classification with zeroth-order optimizers

```bash
bash pipeline/scripts/run_classification_zo.sh --dataset_dir /path/to/dataset --trainer_type <trainer_type> \
  --learning_rate <learning_rate> --load_fp16 <true/false>
```

For example, to reproduce MeZO results on MNLI:

```bash
bash pipeline/scripts/run_classification_zo.sh --dataset_dir data/mnli --trainer_type mezo \
  --learning_rate 5e-7 --load_fp16 true
```

### Classification with first-order optimizers

```bash
bash pipeline/scripts/run_classification_fo.sh --dataset_dir /path/to/dataset --optim <optim> \
  --learning_rate <learning_rate> --bf16 --bf16 true
```

For example, to reproduce AdamW results on MNLI:

```bash
bash pipeline/scripts/run_classification_fo.sh --dataset_dir data/mnli --optim adamw_torch \
  --learning_rate 5e-5 --bf16 --bf16 true
```

## Generation task

### Generation with zeroth-order optimizers

```bash
bash pipeline/scripts/run_generation_zo.sh --dataset_dir /path/to/dataset --trainer_type <trainer_type> \
  --learning_rate <learning_rate> --load_fp16 <true/false>
```

For example, to reproduce MeZO results on SamSum:

```bash
bash pipeline/scripts/run_generation_zo.sh --dataset_dir data/samsum --trainer_type mezo \
  --learning_rate 5e-7 --load_fp16 true
```

### Generation with first-order optimizers

```bash
bash pipeline/scripts/run_generation_fo.sh --dataset_dir /path/to/dataset --optim <optim> \
  --learning_rate <learning_rate> --bf16 --bf16 true
```

For example, to reproduce AdamW results on SamSum:

```bash
bash pipeline/scripts/run_generation_fo.sh --dataset_dir data/samsum --optim adamw_torch \
  --learning_rate 5e-5 --bf16 --bf16 true
```

## On Windows

You can run the corresponding `*.ps1` scripts in the `pipeline/scripts` directory to reproduce the results.
The arguments are the same as the `*.sh` scripts.

## Methods

### Zeroth-order

To run the following methods, make sure use pass these arguments:

| Method      | `trainer_type` | `extra_args`               |
|-------------|----------------|----------------------------|
| MeZO        | `mezo`         |                            |
| ZO-SGD-CONS | `mezo`         | `--zo_subtype zo-sgd-con`  |
| ZO-SGD-SIGN | `mezo`         | `--zo_subtype zo-sgd-sign` |
| ZO-SGD-MMT  | `zosgdmmt`     |                            |
| ZO-Adam     | `zoadam`       |                            |
| R-AdaZO     | `zoadam`       | `--zo_adam_subtype radazo` |
| LOZO        | `lozo`         |                            |
| LOZO-M      | `lozom`        |                            |
| ZO-Muon     | `zomuon`       |                            |
| HiZOO       | `hizoo`        |                            |
| FZOO        | `fzoo`         |                            |

### First-order

To run the following methods: make sure use pass these arguments:

| Method     | `optim`       | `extra_args`      |
|------------|---------------|-------------------|
| AdamW      | `adamw_torch` |                   |
| SGD        | `sgd`         |                   |
| AdamW-LoRA | `adamw_torch` | `--use_lora true` |
