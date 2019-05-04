from setuptools import setup, find_packages

setup(
    name='deepclustering',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/jizongFox/deep-clustering-toolbox',
    license='',
    author='jizong',
    author_email='jizong.peng.1@etsmtl.net',
    description='', install_requires=['numpy', 'torchvision', 'torch', 'Pillow', 'scikit-learn', 'behave', 'requests',
                                      'scikit-image', 'pandas', 'easydict', 'pathlib2', 'matplotlib', 'typing_inspect',
                                      'pytest', 'pytest-cov']
)
