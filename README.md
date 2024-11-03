# CTCoreNet

Neural network for classifying rock clasts in computed tomography (CT) scans of sediment cores.

![GitHub top language](https://img.shields.io/github/languages/top/weiji14/ctcorenet.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Test CTCoreNet](https://github.com/weiji14/ctcorenet/actions/workflows/python-app.yml/badge.svg)](https://github.com/weiji14/ctcorenet/actions/workflows/python-app.yml)
![License](https://img.shields.io/github/license/weiji14/ctcorenet)

![CT scan of sediment core with Ice-Rafted Debris (IRD) clasts highlighted in red](https://dagshub.com/weiji14/ctcorenet/raw/5a59d8f3c7f2d7b9a7bae50f4362b95994c34972/data/train/RS15-LC42_42-188.5-218.5/label_viz.png)

# Getting started

## Quickstart

Launch in [Binder](https://mybinder.readthedocs.io/en/latest/index.html) (Interactive jupyter lab environment in the cloud).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/weiji14/ctcorenet/main)

## Installation

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend
[using conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
to install the dependencies. The conda virtual environment will also be created with
Python and [JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd ctcorenet
    conda env create --file=environment.yml --solver=libmamba

Activate the virtual environment first.

    conda activate ctcorenet

Finally, double-check that the libraries have been installed.

    conda list

## Usage

### Training data preparation

Images are manually labeled by drawing polygons around objects of interest,
e.g. ice rafted debris (IRD) rock clasts. The polygons are drawn using the
[labelme](https://github.com/wkentaro/labelme) GUI program which is launched
from the command-line using:

    labelme

Follow the single image annotation tutorial at
https://github.com/wkentaro/labelme/tree/master/examples/tutorial to draw the
polygons. The polygons are stored in a JSON file, and can be converted to image
masks using:

    labelme_json_to_dataset data/CT_data/CORE_42.json -o data/train/CORE_42

This will produce a folder named CORE_42 containing 4 files:

- img.png
- label.png
- label_names.txt
- label_viz.png

### Running the neural network

The model can be trained using the following command:

    python ctcorenet/ctcoreunet.py

This will load the image data stored in `data/train`, perform the training
(minimize loss between img.png and label.png), and produce some outputs.

More advanced users can customize the training, e.g. to be more deterministic,
running for only x epochs, train on a CUDA GPU using 16-bit precision, etc, like so:

    python ctcorenet/ctcoreunet.py --deterministic=True --max_epochs=3 --accelerator=gpu --devices=1 --precision=16

More options to customize the training can be found by running
`python ctcorenet/ctcoreunet.py --help`.

### Reproducing the entire pipeline

To easily manage the whole machine learning workflow, this project uses the
data version control ([DVC](https://github.com/iterative/dvc/)) library which
stores all the commands and input/intermediate/output data assets used. This
makes it easy to reproduce the entire pipeline using a single command

    dvc pull
    dvc repro

This command will perform all the data preparation and model training steps.
For more information, see https://dvc.org/doc/start/data-pipelines.
