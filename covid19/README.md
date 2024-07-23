# COVID-19

PyTorch-based ML model for detecting COVID-19 based on chest X-ray images. The model classifies images into three categories: Normal, Viral Pneumonia, and COVID-19.

## Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pytorch can be installed from [pytorch-website](https://pytorch.org/get-started/locally/) based on your system configuration.

For example, to install on a linux system with ROCm support, you can run:

```bash
pip3 install torch~=2.3.1 torchvision~=0.18.1 --index-url https://download.pytorch.org/whl/rocm6.0
```

## Install

- Fetch the data from Kaggle - [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) dataset
- You will get `archive.zip` in the folder

Run:

```bash
python tools/prepare_datasets.py archive.zip -d data
```

This will create `data` directory with datasets divided in 3 hospitals (`h1`, `h2` and `h3`) plus additionally a `test` dataset, which will be used to test the produced model.

## Train Model

`covid19.py` is a model that needs to be trained.

Do do the training, execute:

```bash
python covid19.py data/h1 data/h2 data/h3 --model model.pth
```

## Test Model

Inference can be done using `predict.py`:

```bash
python predict.py --model model.pth --image data/test/COVID/COVID-2.png
```
