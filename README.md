# Structured Semidefinite Solvers
---
Welcome to Structured Semidefinite Solvers. This currently barren project will one day host the source code for a number (1 or 2 at least!) of LMI program solvers.

## Contributing
---
To start contributing, first make sure that you have an up-to-date installation of Miniconda (https://www.anaconda.com/docs/getting-started/miniconda/install) and then clone the repository. To create a conda environment with all the required dependencies run the following command:
```
conda env create -f environment.yml
```
in your project directory. Don't worry, these are already in the .gitignore This will initialize the Python virtual environment, but *not* activate it. To activate, simply run

```
conda activate sdp
```