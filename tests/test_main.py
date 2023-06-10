import logging
from pathlib import Path
import pandas as pd

from autods.main import AutoDS
from autods import datasets

ASSETS_FOLDER = Path(__file__).parent.parent.joinpath("assets")


def test_datasets():
    ds_list = []

    datasets = AutoDS.supported_datasets()
    for i, ds in enumerate(datasets):
        assert isinstance(ds.task_names, list) and len(ds.task_names) > 0
        assert isinstance(ds.remote_urls, dict) and len(ds.remote_urls) > 0
        assert isinstance(ds.name, str) and len(ds.name) > 0
        assert isinstance(ds.default_task_name, str) and len(ds.default_task_name) > 0
        assert isinstance(ds.name, str) and len(ds.name) > 0

        ds_list.append(
            {
                "Dataset ID": i,
                "Dataset Name": f"[{ds.name}]({ds.metadata_url})",
                "Default Task": ds.default_task_name,
                "Remote Files": len(ds.remote_urls),
                "Subtasks": ds.task_names,
            }
        )
    assert len(ds_list) > 0
    df = pd.DataFrame(ds_list)
    df.to_markdown(ASSETS_FOLDER.joinpath("DATASET_TABLE.md"), index=False)


if __name__ == "__main__":
    test_datasets()
