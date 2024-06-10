# QSOLenseFinder

QSOLenseFinder is a tool designed to determine whether or not QSOs (Quasi-Stellar Objects) in the DESI (Dark Energy Spectroscopic Instrument) survey are strong gravitational lenses. This repository contains a Python implementation of the model using Keras version 2.13.1 and NumPy version 1.24.3.

## Overview

This project includes a neural network architecture and prediction function to classify QSOs based on their spectra. The model architecture uses a series of convolutional layers followed by fully connected layers to perform binary classification.

### Functions

#### `predict(array)`

Predicts the output using the network architecture defined in the `network` function.

- **Parameters**: 
  - `array` (np.ndarray): Input data array to be used for prediction.
- **Returns**: 
  - `np.ndarray`: Predicted output from the model.
- **Description**: 
  - Initializes the model by calling the `network` function.
  - Loads pre-trained weights from `'new_model_weights_3.h5'`.
  - Uses the model to predict the output based on the input data array.

#### `network(array, lr=1e-3, input_layer=50, n_node_CNN=[100, 100, 100, 100, 100], n_node_FC=[30, 25])`

Defines the neural network architecture for the given input data.

- **Parameters**: 
  - `array` (np.ndarray): Input data array to define the input shape of the network.
  - `lr` (float): Learning rate for the optimizer. Default is `1e-3`.
  - `input_layer` (int): Number of filters in the first Conv1D layer. Default is `50`.
  - `n_node_CNN` (list): List of integers defining the number of filters in each subsequent Conv1D layer. Default is `[100, 100, 100, 100, 100]`.
  - `n_node_FC` (list): List of integers defining the number of nodes in each fully connected (Dense) layer. Default is `[30, 25]`.
- **Returns**: 
  - `tensorflow.keras.models.Model`: Compiled Keras model with the defined architecture.
- **Description**: 
  - The network architecture includes:
    - An input layer that takes the input data.
    - A series of Conv1D and MaxPooling1D layers as specified by the parameters.
    - Flattening of the output from the convolutional layers.
    - A series of fully connected (Dense) layers as specified by the parameters.
    - An output layer with a sigmoid activation function for binary classification.

### Dependencies

- Keras 2.13.1
- NumPy 1.24.3

### Installation

To use this project, ensure you have the required versions of Keras and NumPy installed. You can install them using pip
