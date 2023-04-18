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
A collection of 83 datasets are currently in Stream with easy extension of more possible dataset.
Stream offers convenient management to datasets and sourcing, and feature extraction utilities from
pre-trained models to speed up & evaluate on downstream scenarios.

### Inter-Disciplinary Tasks

- Text (e.g. Yelp)
- Vision (e.g. CIFAR-10, Places-365)

### Feature Vectors

We provide the preprocessed features to download for 83 datasets.
Features from state-of-the-art models CLIP, GPT2, ResNet, and ViT support fast evaluation by training a small downstream model,
as well as multi-modal learning. Our feature vector dataset is a **performant** memory mapped database.

## Install

### Requirements

Stream has been tested on Ubuntu 18.xx and Python 3.10.
For extracting archives the following must be installed in your system and be on Path.

* [patool](https://wummel.github.io/patool/)
* [unrar](https://packages.ubuntu.com/search?keywords=unrar)

**Kaggle Set-up**

Stream uses Kaggle API to download the Kaggle datasets, you need to authenticate the API token before downloading.
Follow the instruction of [Kaggle API: Getting Started: Installation & Authentication](https://www.kaggle.com/docs/api)
to setup Kaggle.

**Feature Extraction**

For extracting features, you will need `ray` for distributed execution. No support without ray as it will be too slow to finish at a reasonable time for many datasets.

`pip install stream[dist]`

**Development**

To contribute, you will need to install with option `[dev]`

`pip install stream[dev]`

### Download Features

TODO

## Examples

### Basic Usage

Easy to use with built-in 83 datasets and downloaded features.
A single dataset can be loaded by specifying `task_id` with custom arguments passed by `datasets`.
The dataset is a Pytorch Dataset class and can be used with [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html) utilities.
The list of supported datasets and their corresponding dataset names and task-ids can be found [HERE](assets/DATASET_TABLE.md).
An example:

```
custom_args = {
    # some datasets are built with different sub-task splits
    'core50': {
        'subset_name': 'object',
        'transform': your_custom_transform,
        'train': False,
    },
}
# load Core50
ds = Stream(root_path, task_id=15, datasets=custom_args)
dl = DataLoader(ds, *args, **kwargs)
```

Alternatively, load all datasets as a ConcatDataset by not specifying `task_id`:

```
# load all datasets
all_ds = Stream(root_path, datasets=custom_args)
dl = DataLoader(all_ds, *args, **kwargs)
```

Load the dataset with extracted features (must download or extract features first):

```
# load dataset as feature vectors
feats_ds = Stream(root_path, task_id=your_task_id, feats_name="clip")
```

### Adding New Dataset

You can fork this repository and add your own dataset.
Follow the example in [test_dataset.py](tests/test_dataset.py) and complete the following abstract properties/methods:

- `metadata_url`: Url to dataset webpage.
- `remote_urls`: File names and download urls. See [here](assets/REMOTE_SOURCES.md) for loading from different types of sources.
- `name`: Dataset name.
- `file_hash_map`: MD5 hash map of data files for verification.
- `dataset_type`: Dataset modality, one of `image` and `text`.
- `default_task_name`: Default subset of the dataset. `none` if no subset.
- `task_names`: List of all subsets. `['none']` if no subset.
- `_process()`: Custom pre-process of downloaded files, e.g. extracting archives.
- `_make_metadata()`: Custom file-to-label mapping.

**NOTICE:** adding new datasets will create a new set of task-ids for loading the datasets.
To check for new task-ids in Stream, run:

```
ds = Stream(root_path)
print(list(enumerate(ds.task_names))
```

In the first use of your dataset, Stream will need to download, extract the files, and making the metadata by:

```
ds = Stream(root_path, task_id=your_task_id, make=True, clean=True)
```

Finally, your custom dataset can be loaded with Basic Usage.

### Extracting Features

Stream supports the following pre-trained models for extracting feature vectors:

* CLIP [Text/Vision]
* GPT2 [Text]
* ResNet [Vision]
* ViT [Vision]

To extract the features for a dataset, pass the feature extractor name when making the dataset.
Specify `batch_size` and `num_gpus` for speeding up extraction.

```
ds = Stream(
    root_path, task_id=your_task_id, feats_name="clip",
    make=True, clean=True, batch_size=128, num_gpus=0.2,
)
# or
ds = your_ds_class(tmp_path, clean=True, **kwargs)
ds.make_features(batch_size, 'cuda', feats_name="clip", clean=True)
```

The feature dataset can be loaded as in Basic Usage.


