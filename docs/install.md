# Installation

## Create a conda environment

Download the latest version of [Miniconda](<https://docs.conda.io/en/latest/miniconda.html>)
for the appropriate OS. Follow the remaining [steps](<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>)
for the appropriate OS version.

Using conda, create a new virtual environment, replacing `<environment_name>` with a name
of your choosing (without spaces):
```text
$ conda create -n <environment_name> python=3.x  # use your preferred 3.9-3.12 version
$ conda activate <environment_name>
```
You can now use ``conda activate <environment_name>`` to enter the environment and
``conda deactivate`` to exit the environment.

## Clone the repository

The CORAL repository can be found [here](https://github.com/NREL/CORAL).

## Dependencies


```{note}
The folowing all asssume you are in your `coral` environment!
```

### Necessary for running CORAL

- Python 3.7+
- marmot-agents
- NumPy
- SciPy
- Matplotlib
- OPENMDAO (>=3.2)

### Development specific dependencies

- jupyterlab
- pandas

