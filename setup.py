from setuptools import setup, find_packages

setup(
    name='deepclustering',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/jizongFox/deep-clustering-toolbox',
    license='',
    author='jizong',
    author_email='jizong.peng.1@etsmtl.net',
    description='',
    install_requires=['pip-tools', 'msgpack', 'numpy', 'torch', 'torchvision', 'Pillow', 'scikit-learn', 'behave',
                      'requests', 'scikit-image', 'pandas', 'easydict', 'pathlib2', 'matplotlib', 'typing_inspect',
                      'tqdm', 'pytest==4.4.0', 'py==1.8.0', 'pytest-remotedata==0.3.1', 'tensorboardX', 'tensorboard',
                      'opencv-python']
)
