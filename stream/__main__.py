import argparse
from pathlib import Path

from stream.main import Stream

args = argparse.ArgumentParser()
args.add_argument("--action", choices=["make", "make_feats_clip"])
args.add_argument("--root_path", type=Path, required=True)
args.add_argument("--clean", action="store_true")
_args = args.parse_args()
root_path = _args.root_path
if _args.action == "make":
    stream = Stream(root_path=root_path, make=True, clean=_args.clean)
    stream.verify()
