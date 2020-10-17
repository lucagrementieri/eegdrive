import re
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).parent.resolve()


def read(*parts):
    with here.joinpath(*parts).open() as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('unable to find version string')


with (here / 'README.md').open(encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='eegdrive',
    version=find_version('eegdrive', '__init__.py'),
    description='EEG classification',
    long_description=long_description,
    author='Luca Grementieri',
    author_email='luca.grementieri@ens-paris-saclay.fr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Healthcare company',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Healthcare :: EEG',
        'License :: Apache 2.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='eeg classification driving neural network',
    packages=find_packages(exclude=['build', 'data', 'dist', 'docs', 'tests']),
    python_requires='>=3.7',
    install_requires=[
        'h5py >= 2.10',
        'numpy >= 1.19',
        'scikit-learn >= 0.23',
        'scipy >= 1.5',
        'tqdm >= 4.50',
        'torch >= 1.6',
    ],
)
