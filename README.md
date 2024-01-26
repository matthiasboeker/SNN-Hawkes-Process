# SNN-Hawkes-Process

## Overview
This repository contains Python scripts for implementing and analyzing a Spiking Neural Network (SNN) integrated with a Hawkes Process model. The primary focus is on predicting spike trains, analyzing neural interactions, and optimizing neural network parameters.

## Contents
1. `modelling_with_hawkles.py` - Contains the `HawkesProcessModel__` class used for modelling the Hawkes process in a neural network.
2. `main.py` - The main script that loads spike train data, initializes the Hawkes process model, computes log-likelihoods, and saves parameter matrices. It also generates heatmaps of alpha parameters for correct and wrong predictions.
3. `neural_network_training.py` - Implements a neural network using leaky integrate-and-fire (LIF) neurons and optimizes it using an evolutionary algorithm (CMA-ES).
4. `hawkes_process_models.py` - Defines several variations of the Hawkes Process model (`HawkesProcessModel`, `HawkesProcessModel_`, `HawkesProcessModelSGD`, `HawkesProcessModel__`) for different analytical and optimization approaches.

## Installation
To run these scripts, you need to have Python installed along with the following libraries:
- numpy
- torch
- matplotlib
- seaborn
- cma
- sklearn

You can install these dependencies using pip:
```bash
pip install numpy torch matplotlib seaborn cma sklearn
```

## Usage

### Modelling with Hawkes Process
To use the Hawkes process model, import the `HawkesProcessModel__` class from `modelling_with_hawkles.py` and initialize it with the appropriate parameters.

### Running the Main Script
Execute `main.py` to process spike train data, perform model computations, and generate heatmaps. Ensure the data paths in the script are correctly set up.

### Training the Neural Network
Run `neural_network_training.py` to train the neural network using the evolutionary algorithm. This script also evaluates the network's performance and saves the results.

### Exploring Hawkes Process Models
The `hawkes_process_models.py` file contains different implementations of the Hawkes process model. Use these models as needed for your specific analysis.
