
## Loading Dataset Files from Different Sources

### Kaggle

Download using Kaggle API `kaggle datasets download -d {path-to-dataset}` or `kaggle competitions download -c {path-to-dataset}`
for competition datasets

Example: Apparel, Aptos2019

```
remote_urls = {
    "apparel-images-dataset.zip": "kaggle datasets download -d trolukovich/apparel-images-dataset",
}
remote_urls = {
    "aptos2019-blindness-detection.zip": "kaggle competitions download -c aptos2019-blindness-detection",
}
```

### Publicly accessible links (URL, HTTP, FTP, Google Drive)

Example: Stanford Cars

```
remote_urls = {
    "cars_train.tgz": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
    "cars_test.tgz": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
    "car_devkit.tgz": "http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
    "cars_test_annos.mat": "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
}
```

**Google Drive**

Put the entire link `https://drive.google.com/uc?export=download&id={download_file_id}` or file id `{download_file_id}`
in `remote_urls`.

Example: Textures

```
remote_urls = {
    "Splited.zip": "13LBYN6eTfV9G9xdgZtdpNHrXSA8mpv-2",
}
```

### Local files

Set the values in `remote_urls` to None to load from local disk.

```
remote_urls = {
    "your-local-file": None,
}
```

### Github repository

Put the name of the file as the name of repository.

Example: Plantdoc

```
remote_urls = {
    "plantdoc": "https://github.com/pratikkayal/PlantDoc-Dataset.git",
}
```


## Processing Downloaded Files

Stream dataset loads the image dataset from individual image files similar to [PyTorch ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).
You will need to extract from the downloaded files and create a metadata mapping from image path to the label.
For text datasets, the metadata should be a mapping from text input to the label.

If the downloaded data files are archive files (e.g. `.zip`, `.tar.gz`, `.rar`), 
you can extract them in the dataset's `_process()` function by:

```
from stream.utils import extract, is_archive

def _process(self, raw_data_dir: Path):
    for archive in self.remote_urls.keys():
        archive_path = raw_data_dir.joinpath(archive)
        if is_archive(archive_path):
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)
```

If the downloaded images are stored in structured files like 
`.h5`, `.pt`, `.mat`, and `.npy`, you need to extract the images and save as separate image files.
See [galaxy10.py](/stream/datasets/galaxy10.py) for an example.