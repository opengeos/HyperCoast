# Installation

## Install from PyPI

**hypercoast** is available on [PyPI](https://pypi.org/project/hypercoast/). To install **hypercoast**, run this command in your terminal:

```bash
pip install hypercoast
```

## Install from conda-forge

**hypercoast** is also available on [conda-forge](https://anaconda.org/conda-forge/hypercoast). If you have
[Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer, you can install hypercoast using the following command:

```bash
conda install -c conda-forge hypercoast
```

Alternatively, you can create a new conda environment and install hypercoast in the new environment. This is a good practice because it avoids potential conflicts with other packages installed in your base environment.

```bash
conda install -n base mamba -c conda-forge
conda create -n hyper python=3.11
conda activate hyper
mamba install -c conda-forge hypercoast
```

## Install from GitHub

To install the development version from GitHub using [Git](https://git-scm.com/), run the following command in your terminal:

```bash
pip install git+https://github.com/opengeos/hypercoast
```
