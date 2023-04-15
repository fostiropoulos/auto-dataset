import shutil
import urllib.request
from pathlib import Path

import gdown
from git import Repo
from pyunpack import Archive

extensions = {
    "",  #: extract_empty,
    ".rar",  #: extract_rar,
    ".gz",  #: extract_tar,
    ".zip",  #: extract_zip,
    ".tgz",  #: extract_tar,
}


def isImageType(t):
    return hasattr(t, "im")


def extract(file_path: Path, save_dir: Path):
    extract_dir = save_dir.joinpath(file_path.stem.split(".")[0])
    extract_dir.mkdir(exist_ok=True, parents=True)
    Archive(file_path.as_posix()).extractall(extract_dir.as_posix())


def is_archive(p: Path):
    return p.suffix.strip() in extensions and p.is_file()


def download_file(url: str, save_dir: Path, filename: str):
    save_dir.mkdir(exist_ok=True, parents=True)
    if url is None:
        save_path = save_dir.joinpath(filename)
        assert save_path.exists(), f"Could not find local file {filename}."
    elif "http" not in url and "ftp" not in url and "kaggle" not in url:
        save_path = save_dir.joinpath(filename)
        gdown.download(
            id=url,
            output=save_path.as_posix(),
            quiet=False,
            use_cookies=False,
            fuzzy=True,
        )
    elif "google" in url:
        save_path = save_dir.joinpath(filename)
        gdown.download(
            url=url,
            output=save_path.as_posix(),
            quiet=False,
            use_cookies=False,
            fuzzy=True,
        )
    elif "kaggle" in url:
        import kaggle

        kaggle.api.authenticate()
        *args, dataset_type, dataset_name = url.split(" ")
        if dataset_type == "-d":
            kaggle.api.dataset_download_files(dataset_name, path=save_dir)
        elif dataset_type == "-c":
            kaggle.api.competition_download_files(dataset_name, path=save_dir)

    elif url.endswith(".git"):
        repo = Repo.clone_from(url, save_dir.joinpath(filename))
        assert (
            Path(repo.working_dir).name == filename
        ), f"Different dataset {filename} with repository working dir"
    else:
        save_path = save_dir.joinpath(filename)
        urllib.request.urlretrieve(url, save_path)
