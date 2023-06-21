# Welcome to NeuralOperatorStarForm's documentation

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    main.py                     # The main script that runs the neural operator.
    input.py                    # The configuration file.
    sweep.py                    # The script that runs the sweeps.
    src/
        dissipative_utils.py    # Functions for the markov neural operator encoder.
        GenerateSweep.py        # Script to generate a sweep config file for wandb.
        meta_dataset.py         # pytorch dataset class that allows the passing of metadata.
        networkUtils.py         # Functions for the neural networks.
        train.py                # Class to train the neural operator.
        utils.py                # Utility functions for preparing the dataset.
