# EEG Drive

Classification of EEG recorded during driving action execution.

## Getting Started

### Prerequisites

In order to run the code you need to have Python 3.6 or above installed.

### Installing

You can install the package on MacOS/Linux with the following commands:
```
git clone https://github.com/lucagrementieri/eegdrive.git
cd eegdrive
python3 setup.py bdist_wheel
python3 -m pip install -r requirements.txt
python3 -m pip install --no-index --find-links=dist eegdrive
```

The commands above install a Python package named `eegdrive` on your machine.
If the package has been successfully installed the command
```
import eegdrive
```
should succeed without any error when run in a Python console.

If you don't install the package, you will be able to use it only from its root directory.

## Usage

EEGDrive comes with a handy command line interface (CLI).
It can be interrogated by executing the package with 
```
python3 -m eegdrive <command>
```

You can access CLI help page with
```
python3 -m eegdrive --help
```

The available commands are:
- `ingest`: preprocess raw data in a HDF5 file and export every episode in a
 compressed `npz` file.
- `train`: train the classification model on ingested episodes.

Every command has its separate help page that can be visualized with
```
python3 -m eegdrive <command> --help
```

### Data ingestion
The `ingest` command takes two required arguments:
- `data-path`: the path to the HDF5 file of an acquisition session;
- `output-dir`: the path of the directory where ingested episodes are stored.

An example of usage is
```
python3 -m eegdrive ingest data/Subject-1.hdf5 data/episodes1
```

### Classifier training
The `train` command takes a required argument and several optional arguments:
- `dataset-dir`: the path to the directory of ingested episodes;
- `--runs-dir` (default `./runs`): the path of the directory where feature extractor random weights are saved;
- `--filters` (default `100`): the number of filters per input channel in every convolutional layer;
- `--label-type` (default `action`): `action` if the task is action classification, 
`preparation` if the model should be trained for action prediction.
- `--seed` (default `42`): random seed used for model initialization and selection of test examples.

An example of usage for action classification is
```
python3 -m eegdrive train data/episodes1
```
while for action prediction the command becomes
```
python3 -m eegdrive train data/episodes1 --label-type preparation
```