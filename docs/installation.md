# Installation

## Install using uv

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and project manager, written in Rust. Follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/) on the uv website to install uv on your computer. Once uv is installed, you can install **hypercoast** using the following command:

```bash
uv venv
uv pip install hypercoast jupyterlab
```

To run JupyterLab, use the following command:

```bash
uv run jupyter lab
```


## Install from PyPI

**hypercoast** is available on [PyPI](https://pypi.org/project/hypercoast/). To install **hypercoast**, run this command in your terminal:

```bash
pip install hypercoast
```

HyperCoast has some optional dependencies that are not installed by default, such as cartopy, earthaccess, mapclassify, and pyvista. To install all optional dependencies all at once, run the following command:

```bash
pip install "hypercoast[extra]"
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

To install the optional dependencies, run the following command:

```bash
mamba install -c conda-forge cartopy earthaccess mapclassify pyvista trame-vtk trame-vuetify
```

## Install from GitHub

To install the development version from GitHub using [Git](https://git-scm.com/), run the following command in your terminal:

```bash
pip install git+https://github.com/opengeos/hypercoast
```
