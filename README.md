# Unisound Chinese Medical Named Entity Recognition

## Quick links

- [Results](#results)
- [Setup](#setup)
  - [Install dependencies](#install-dependencies)
  - [Preprocess the datasets](#preprocess-the-datasets)
- [Quick Start](#quick-start)

## Results

We do experiment on Ubuntu 16.04 with Intel Xeon CPU E5-2620 @3.2GHz and GeForce GTX 1080 Ti 12GB. Python and Anaconda version are 3.7.11 and 4.10.1 respectively.

Multi-label prediction:
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

Disease-label prediction:
```txt
 -------------------------------------
|     DISEASE     | p=89% r=87% f=88% |
 -------------------------------------
```

## Setup

### Install dependencies

Please install all the dependency packages using the following command:

```bash
pip install -r requirements.txt
```

### Preprocess the datasets

Please put `train.txt` and `test.txt` in `/data/raw/` with formatting as followed:

```txt
...
预 B-PURPOSE
防 E-PURPOSE
和 O
治 B-PURPOSE
疗 E-PURPOSE
癌 B-CONDITAION
症 I-CONDITAION
化 I-CONDITAION
疗 E-CONDITAION
引 O
起 O
...
```

Directory `/data/dis/` contains data with only DISEASE label:

```txt
...
儿 O
童 O
的 O
支 B-DISEASE
气 I-DISEASE
管 I-DISEASE
哮 I-DISEASE
喘 E-DISEASE
...
```

## Quick Start

The following commands can be used to run our pre-trained model on `/data/`. You can also fine tune our pre-trained model with extra dataset with `--mode=finetune`:

```bash
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

Or only predict DISEASE label with light and efficient crf model by using following command:

```bash
python core/run_crf.py
```
