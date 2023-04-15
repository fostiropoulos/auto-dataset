#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="stream",
    version="1.0",
    description="Stream-Dataset",
    author="Iordanis Fostiropoulos",
    author_email="fostirop@usc.edu",
    python_requires=">3.10",
    long_description=open("README.md").read(),
    url="https://deep.usc.edu/",
    packages=find_packages(),
    zip_safe=False,

    install_requires=[
        "patool",
        "GitPython",
        "pandas",
        "scipy",
        "torch",
        "scikit-learn",
        "transformers",
        "Pillow",
        "idx2numpy",
        "patool @ https://github.com/wummel/patool/archive/refs/heads/master.zip",
        "pyunpack",
        "gdown",
        "h5py",
        "numpy",
        "kaggle",
        "unrar",
        "lmdb",
        "torch",
        "setproctitle",
        "torchvision",
        "tqdm","tabulate"
    ],
    extras_require={
        "dev": ["mypy", "pytest", "pylint", "flake8", "black","types-requests"],
        "dist": ["ray"],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu113",
    ],
)
