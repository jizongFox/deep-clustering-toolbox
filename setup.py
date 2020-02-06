from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="deepclustering",
    entry_points={
        "console_scripts": [
            "viewer=deepclustering.viewer.Viewer:main",
            "clip_screencapture=deepclustering.postprocessing.clip_images:call_from_cmd",
            "report=deepclustering.postprocessing.report2:call_from_cmd",
        ]
    },
    version="0.0.2",
    packages=find_packages(
        exclude=["playground", ".data", "script", "test", "runs", "config"]
    ),
    url="https://github.com/jizongFox/deep-clustering-toolbox",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jizong Peng",
    author_email="jizong.peng.1@etsmtl.net",
    install_requires=[
        "msgpack",
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "scikit-learn",
        "behave",
        "requests",
        "scikit-image",
        "pandas",
        "easydict",
        "matplotlib",
        "tqdm==4.32.2",
        "py==1.8.0",
        "pytest-remotedata==0.3.1",
        "tensorboardX",
        "tensorboard",
        "opencv-python",
        "medpy",
        "pyyaml",
        "termcolor",
        "gpuqueue",
        "gdown",
    ],
)
