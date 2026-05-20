# Dataset processing module

Use `load_dataset` to load a dataset from Hugging Face's Datasets library.
The script will automatically preprocess the dataset and save it in a format suitable for the pipeline.

Make sure you enter your HuggingFace token into `HF_TOKEN.txt`.

Then you can the command below to load a dataset:

```bash
python dataset_processing/load_dataset.py --dataset <DATASET_NAME>
```

-------

Datasets to download:

- MNLI: `python dataset_processing/load_dataset.py --dataset mnli`
- Emotion: `python dataset_processing/load_dataset.py --dataset emotion`
- SamSum: `python dataset_processing/load_dataset.py --dataset samsum`
- MATH: `python dataset_processing/load_dataset.py --dataset math`

To download all, run:

```bash
bash dataset_processing/scripts/download_all.sh
```

or (for Windows PowerShell):

```bash
.\dataset_processing\scripts\download_all.ps1
```
