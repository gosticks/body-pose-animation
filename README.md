# realtime-body-tracking

## Installation

This project is using python version **3.6.0**. It is also recommended to use anaconda for managing your python environments.

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

## Links

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
