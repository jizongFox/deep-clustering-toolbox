import pathlib

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="deepclustering",
    version="1.0.0",
    packages=find_packages(),
    url="https://github.com/jizongFox/deep-clustering-toolbox",
    license="MIT    ",
    author="Jizong Peng",
    author_email="jizong.peng.1@etsmtl.net",
    description="",
    install_requires=[
        "pip-tools",
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
        "pathlib2",
        "matplotlib",
        "typing_inspect",
        "tqdm==4.32.2",
        "pytest==4.4.0",
        "py==1.8.0",
        "pytest-remotedata==0.3.1",
        "tensorboardX",
        "tensorboard",
        "opencv-python",
        "medpy",
        "pyyaml",
        "termcolor"
    ],
    entry_points = {
        "console_script":[
            "viewer=deepclustering.viewer.Viewer:main",
        ]
    },
)
