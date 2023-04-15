# Stream Dataset

**Stream** (of consciousness? 🤔 ) Training on a large sequence of multi-disciplinary datasets. This repository implements the logic for processing and managing a large sequence of datasets.

Stream can automatically download and process dataset from:
* Kaggle
* Publicly accessible via a URL, HTTP, FTP, or Google Drive link
* Locally stored
* git repository

It can be extended to any dataset where only the logic of extracting and processing the dataset is required via sub-classing and method over-writing of a [Dataset](stream/dataset.py) class.

This dataset is a contribution of our paper Batch-Model-Consolidation at CVPR 2023. Please cite:

```

```

## Why Stream?

Stream provides a method to train on interdisciplinary tasks by projecting all datasets on the same dimension.

### Inter-Disciplinary Tasks

- Text
- Vision

### Feature Vectors

- We use the feature vectors from different backbone

## Download Features

We provide the preprocessed features to download for 84 datasets.

## Install

### Requirements

Currently tested on a Ubuntu 18.xx >

Python 3.10 >

For extracting archives the following must be installed in your system and be on Path.

* [patool](https://wummel.github.io/patool/)
* [unrar](https://packages.ubuntu.com/search?keywords=unrar)

**Feat Extraction**

For extracting features, you will need `ray` for distributed execution. No support without ray as it will be too slow to finish at a reasonable time for many datasets.

`pip install stream[dist]`

**Development**

To contribute, you will need to install with option `[dev]`

`pip install stream[dev]`

## Supported Feature Extractors

* CLIP [Text/Vision]
* GPT2 [Text]
* ResNet [Vision]
* ViT [Vision]

## Supported Datasets

The list of the supported dataset is below. You can fork this repository and add your own dataset.

[DATASETS](assets/DATASET_TABLE.md)

# TODO
[INSTRUCTIONS](assets/INSTRUCTIONS.md)

## Usage Examples

### TODO

