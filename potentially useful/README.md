# Potentially Useful Files
More description and comments in individual files

## Base directory contains files for variable number of stations architecture

### data_pipeline.py
This file defines the generator functions which are the first stage of the data input pipeline and are responsible for
opening up the mag files and doing some preprocessing.

### mag_nn
this file has the MagNN class and subclasses which define model architectures, training loops and other convenience functions

### run_model.py
This file instantiates a MagNN object and either loads an already trained model or trains a new model

### modules.py
this file contains modules which make defining architectures more convenient

### models directory
this directory has save files for MagNN-style models

## closest stations architecture

### create_single_station_dataset.py
Creates a dataset to be used for the closest stations architecture.

### single_station_experiment.py
Defines, trains, saves a model with closest stations architecture

### models directory
saved models

