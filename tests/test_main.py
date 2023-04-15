import logging
from pathlib import Path
import pandas as pd

from stream.main import Stream
from stream import datasets

ASSETS_FOLDER = Path(__file__).parent.parent.joinpath("assets")


def test_datasets():
    ds_list = []

    datasets = Stream.supported_datasets()
    for ds in datasets:
        assert isinstance(ds.task_names, list) and len(ds.task_names) > 0
        assert isinstance(ds.remote_urls, dict) and len(ds.remote_urls) > 0
        assert isinstance(ds.name, str) and len(ds.name) > 0
        assert isinstance(ds.default_task_name, str) and len(ds.default_task_name) > 0
        assert isinstance(ds.name, str) and len(ds.name) > 0

        ds_list.append(
            {
                "Dataset Name": f"[{ds.name}]({ds.metadata_url})",
                "Default Task": ds.default_task_name,
                "Remote Files": len(ds.remote_urls),
                "Subtasks": ds.task_names,
            }
        )
    assert len(ds_list) > 0
    df = pd.DataFrame(ds_list)
    df.to_markdown(ASSETS_FOLDER.joinpath("DATASET_TABLE.md"), index=False)


def test_stream():
    logging.basicConfig(level=logging.INFO)

    root_path = Path.home().joinpath("stream_ds")

    # datasets = Stream.find_datasets(PACKAGE_DATASET_DIR)
    # ray.init(address = "auto")
    s = Stream(root_path, task_id=0)
    s.export_feats(Path.home().joinpath("stream_ds_feats"))
    pass

    # s = Stream(root_path, returnt_feats=False, exclude=["imdb"])
    # # s.make_features(
    # #     batch_size=128, num_gpus=0.1, clean_make=True, feature_extractor="resnet"
    # # )
    # s.make_features(
    #     batch_size=64, num_gpus=0.2, clean_make=True, feature_extractor="vit"
    # )
    # repeat_ds_vit = ["emnist", "inaturalist", "places365"]
    # # repeat_ds = [
    # #     "inaturalist"
    # # ]

    # remotes = []
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

    # for ds in datasets:
    #     if ds.__name__.lower() in repeat_ds_vit:
    #         logging.warning(f"Making {ds.__name__}")
    #         # Stream.make_dataset_features(ds, root_path, 2048, True, "resnet")

    #         # Stream.make_dataset_features(ds, root_path, 512, True, "vit")
    #         remotes.append(
    #             ray.remote(num_gpus=1, max_calls=1)(
    #                 Stream.make_dataset_features
    #             ).remote(ds, root_path, 512, True, "vit")
    #         )
    #         # break
    # ray.get(remotes)

    # s.make_dataset_features(dss[0], root_path, 128, clean_make=True)
    # s = Stream(root_path, task_id=0, return_feats=True)
    # for i, task in enumerate(Stream.find_datasets(DATASET_CLASS_FOLDER)):
    #     s = Stream(root_path, make=True, return_feats=False)

    # Stream.make_dataset(task, kwargs=dict(root_path=root_path))
    # s = Stream(root_path, return_feats=False, task_id=i)
    # s.verify()
    # s.assert_dataset()


if __name__ == "__main__":
    test_stream()
