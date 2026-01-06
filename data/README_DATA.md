# Italian Parkinson Voice & Speech dataset

The recipes expect the raw dataset to be placed under `data/raw/italian_parkinson/`.

Download from the Hugging Face mirror of the IEEE Dataport release:

```
wget "https://huggingface.co/datasets/birgermoell/Italian_Parkinsons_Voice_and_Speech/resolve/main/italian_parkinson/Italian%20Parkinson's%20Voice%20and%20speech.zip?download=true" -O dataset.zip
unzip dataset.zip -d data/raw/italian_parkinson
```

The folder structure should look like:

```
data/raw/italian_parkinson/
 ├── 15 Young Healthy Control/
 ├── 22 Elderly Healthy Control/
 └── 28 People with Parkinson's disease/
```

After downloading, run `python scripts/prepare_manifests.py --data_root data/raw/italian_parkinson --out_dir data/manifests --split_by speaker` to create manifests for training.
