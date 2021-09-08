# Unisound Chinese Medical Named Entity Recognition

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=Python&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. Codes are all writen with **Python**.

Source code of Unisound-NER project. Some important information has been ignored.

## Quick links

- [Results](#results)
- [Setup](#setup)
  - [Install dependencies](#install-dependencies)
  - [Preprocess the datasets](#preprocess-the-datasets)
- [Quick Start](#quick-start)

## Results

We do experiment on Ubuntu 16.04 with Intel Xeon CPU E5-2620 @3.2GHz and GeForce GTX 1080 Ti 12GB. Python and Anaconda version are 3.7.11 and 4.10.1 respectively.

```txt
 -------------------------------------------------
|      TOTAL      | p=81.388% r=83.226% f=82.297% |
|-----------------|-------------------------------|
|     DISEASE     | p=82.809% r=91.200% f=86.802% |
|     PURPOSE     | p=76.190% r=86.486% f=81.013% |
|   PERSON_GROUP  | p=66.667% r=45.455% f=54.054% |
|    CONDITION    | p=80.392% r=63.077% f=70.690% |
|     SUMPTOM     | p=83.784% r=71.264% f=77.019% |
| INAPPLICABILITY | p=100.00% r=100.00% f=100.00% |
 -------------------------------------------------
```

## Setup

### Install dependencies

Please install all the dependency packages using the following command:

```bash
pip install -r requirements.txt
```

### Preprocess the datasets

Please put `train.txt` and `test.txt` in `/data/` with formatting as followed:

```txt
Chinese-token B-label
Chinese-token E-label
Chinese-token O
Chinese-token B-label
Chinese-token E-label
Chinese-token B-label
Chinese-token I-label
Chinese-token I-label
Chinese-token E-label
Chinese-token O
Chinese-token O
...
```

## Quick Start

The following commands can be used to run our pre-trained model on `/data/`.

```bash
# Download pre-trained model RoBERTa from Google Drive: https://drive.google.com/file/d/1IRUAbf-ML2mekK6Ysmyav78u_Y03wMnh/view?usp=sharing
unzip roberta.zip
mv roberta model
python core/bert.py \
    --save \
    --mode=train \
    --pretrained_bert_path=model/roberta_wwm_ext_large \
    --saved_path=model \
    --model_path=model \
    --trainset_path="['./data/raw/train.txt']" \
    --testset_path=./data/raw/test.txt \
    --cuda=0 \
    --batch_size=2

# Or download our pre-trained model from Google Drive to fine-tune with extra dataset:https://drive.google.com/file/d/1HAO4cqc0lYR7e54GUM2uebWXYXj5hsS6/view?usp=sharing
unzip model.zip
python core/bert.py \
    --save \
    --mode=finetune \
    --pretrained_bert_path=model/roberta_wwm_ext_large \
    --saved_path=model \
    --model_path=model \
    --trainset_path="['./data/raw/train.txt']" \
    --testset_path=./data/raw/test.txt \
    --cuda=0 \
    --batch_size=2
```

The output files will be stored in `/pred/`. Please use following command to post-process data:

```bash
python core/compose.py \
    --path={OUTPUT FILE}
```

You may want to evaluate other results with following command:

```bash
mv {OTHER RESULT} BIEO.txt
python core/evaluate.py
```
