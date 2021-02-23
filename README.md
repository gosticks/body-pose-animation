# realtime-body-tracking

## Installation

This project is using python version **3.6.0**. It is also recommended to use anaconda for managing your python environments.

### Downloading SMPL models

The SMPL models can be downloaded from [this link](http://smplify.is.tue.mpg.de/downloads). Alternatively just use the following script:

```bash
./setup.sh
```

This should copy and rename the SMPL model to the correct folders. Either way the resulting folder structure should look like this (ignore smplx if you don't intend to use it):

```bash
└── models
    ├── smpl
    │   ├── SMPL_FEMALE.npz
    │   ├── SMPL_MALE.npz
    │   └── SMPL_NEUTRAL.pkl
    └── smplx
        ├── SMPLX_FEMALE.npz
        ├── SMPLX_FEMALE.pkl
        ├── SMPLX_MALE.npz
        ├── SMPLX_MALE.pkl
        ├── SMPLX_NEUTRAL.npz
        └── SMPLX_NEUTRAL.pkl
```

### VPoser
To use `bodyPrior` in the configuration please download vposer and plate it into `./vposer_v1_0` directory in the project root. Vposer can be downloaded from this link after creating an account with SMPL-X [link](https://psfiles.is.tuebingen.mpg.de/downloads/smplx/vposer_v1_0-zip)

### Mesh intersection 
To use `intersectLoss` in the configuration please pull the [github](https://github.com/gosticks/torch-mesh-isect) repo. This repo is patched to run on the newer versions of pytorch. 
Note: It only runs for Linux based operating systems. We had troubles getting it to work on Windows.

### Conda Environment

Create a new conda env by typing

```bash
conda create --name tum-3d-proj python=3.6
```

When using VSCode the IDE should automatically use the python interpreter created by conda.

### Pip packages

The required pip packages are stored within the `requirements.txt` file and can be installed via

```bash
pip install -r requirements.txt
```

### Exporting imported packages to pip

Since pip is trash and does not save imported packages in a file within the project you have to force it to do so.

```bash
pip freeze > requirements.txt
```

## Usage

The project provides sample examples to the usage of the project. These examples can be directly run after installing all required packages. Per default the input data is expected to be located in the `./samples/` folder. We would recommend you to get all the frames from the source video in png format and pass the video through OpenPose to get the results exported in .json format if you want to try your own samples.

### Input data

The input is expected to be OpenPose keypoints and output images (required for previews). These files should be placed to the samples folder the foldername and the name format of the samples can be configured within the `config.yaml`.

### Selecting config file

By default the config is expected at `./config.yaml` this can be changed to any value by setting the `CONFIG_PATH` environment variable.

```bash
CONFIG_PATH=/path/to/myconfig.yaml python example_fit.py
```

## Links

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
