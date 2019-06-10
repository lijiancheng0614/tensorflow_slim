# TensorFlow Slim

Code for training and evaluating several widely used Convolutional Neural Network (CNN) image classification models using [TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

Forked from https://github.com/tensorflow/models/tree/master/research/slim


## Requirements

1. Install the [TensorFlow library](https://www.tensorflow.org/install/).

2. Data. For example,

    ```
    data
    +-- images
    |   +-- 1.jpg
    |   +-- ...
    +-- labels.txt
    +-- list_train.txt
    +-- list_val.txt
    ```


## Train

Training a model from scratch:

```bash
python train.py \
    --dataset_dir=data/train \
    --labels_to_names_path=data/labels.txt
```

Fine-tuning a model from an existing checkpoint `checkpoints/inception_v3.ckpt`:

```bash
python train.py \
    --dataset_dir=data/train \
    --labels_to_names_path=data/labels.txt \
    --checkpoint_path=checkpoints/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
```


## Evaluation

```bash
python eval.py \
    --dataset_dir=data/val
```


## Inference

```bash
python infer.py \
    --images_dir=data/images \
    --labels_path=data/labels.txt
```
