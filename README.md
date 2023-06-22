<!-- headline -->
# NeuralOperatorStarForm
Train a fourier neural operator to simulate MHD turbulence in a star-forming molecular cloud. Here is a link to the [docs](https://kpoletti.github.io/NeuralOperatorStarForm/)

Currently under development. Software is configured to be used with four datasets:
1. Turbulent Molecular Cloud - 2-D column density of a turbulent molecular cloud with varying magnetic field strength. It is self gravitating.
1. Voricity Navier-Stokes - 2-D vortcity data from a Navier-Stokes simulation produce by [Neural Operator team](https://github.com/neuraloperator/neuraloperator)
1. Gravitional Collapse - 2-D column density of a gravitationally collapsing spherically molecular cloud
1. [CATS MHD](https://users.flatironinstitute.org/~bburkhart/data/CATS/MHD/machine_learning/) - Ideal MHD in a periodic box without gravity. The data is produced by [Burkhart et al. 2009](http://adsabs.harvard.edu/abs/2009ApJ...693..250B), [Cho and Lazarian 2003](http://adsabs.harvard.edu/abs/2003MNRAS.345..325C), and [Burkhart et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...905...14B/abstract)

<!-- toc -->
## Table of Contents
<!-- - [Installation](#installation) -->
- [Usage](#usage)
- [Dependencies](#dependencies)
<!-- - [Contributing](#contributing)
- [License](#license) -->

<!-- installation -->
<!-- ## Installation
To install the package, run the following command in the root directory of the repository:
```bash
pip install -e .
``` -->

<!-- usage -->

## Usage
To train a model, first configure `input.py` to the desired dataset, network, and hyperparameters. Then run the following command:
```bash
python main.py
```
Note this model can be expensive to train. To reduce the dataset resolution change `poolKernel` and `poolStride` in `input.py` to a larger value divisible by the total resolution `S`.

## Dependencies
- [PyTorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://www.scipy.org/)
- [wandb](https://wandb.ai/site)
- [pandas](https://pandas.pydata.org/)
- [neural operator](https://github.com/neuraloperator/neuraloperator)
