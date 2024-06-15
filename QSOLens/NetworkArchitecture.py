import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Activation, Dense, Dropout
import keras

def network_classifier(array, input_layer = 50, n_node_CNN = [100, 100, 100, 100, 100], n_node_FC = [30, 25]):
    """
    Defines the neural network architecture for the given input data.
    
    Parameters:
    array (np.ndarray): Input data array to define the input shape of the network.
    lr (float): Learning rate for the optimizer (default is 1e-3).
    input_layer (int): Number of filters in the first Conv1D layer (default is 50).
    n_node_CNN (list): List of integers defining the number of filters in each subsequent Conv1D layer (default is [100, 100, 100, 100, 100]).
    n_node_FC (list): List of integers defining the number of nodes in each fully connected (Dense) layer (default is [30, 25]).
    
    Returns:
    tensorflow.keras.models.Model: Compiled Keras model with the defined architecture.
    
    Description:
    The network architecture includes:
    - An input layer that takes the input data.
    - A series of Conv1D and MaxPooling1D layers as specified by the parameters.
    - Flattening of the output from the convolutional layers.
    - A series of fully connected (Dense) layers as specified by the parameters.
    - An output layer with a sigmoid activation function for binary classification.
    """
    
    input_shape = (len(array.iloc[1]), 1)  # Assuming each element in X_train is a 1D array
    
    # Input layer
    inputs = Input(shape = input_shape)

    x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    for j, k in enumerate(n_node_CNN): 
        x = Conv1D(k, kernel_size=5, activation='relu')(x)
        if j<(len(n_node_CNN)): 
            x = MaxPooling1D(pool_size=2)(x)

    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully Connected Layer
    for i, n in enumerate(n_node_FC): 
        x = Dense(n, activation='relu')(x)

    
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    mod1 = Model(inputs=inputs, outputs=outputs)

    # Define optimizer and metrics
    initial_learning_rate = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    metrics = keras.metrics.F1Score(
        average=None, threshold=0.5, name='f1_score', dtype=None
    )

    # Compile the model
    mod1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', metrics, 'AUC'])

    return mod1

def network_z_finder(input_layer = 50, n_node_CNN = [100, 100, 100, 100, 100], n_node_FC = [30, 25]):
    """
    Defines the neural network architecture for the given input data.
    
    Parameters:
    array (np.ndarray): Input data array to define the input shape of the network.
    lr (float): Learning rate for the optimizer (default is 1e-3).
    input_layer (int): Number of filters in the first Conv1D layer (default is 50).
    n_node_CNN (list): List of integers defining the number of filters in each subsequent Conv1D layer (default is [100, 100, 100, 100, 100]).
    n_node_FC (list): List of integers defining the number of nodes in each fully connected (Dense) layer (default is [30, 25]).
    
    Returns:
    tensorflow.keras.models.Model: Compiled Keras model with the defined architecture.
    
    Description:
    The network architecture includes:
    - An input layer that takes the input data.
    - A series of Conv1D and MaxPooling1D layers as specified by the parameters.
    - Flattening of the output from the convolutional layers.
    - A series of fully connected (Dense) layers as specified by the parameters.
    - An output layer with a sigmoid activation function for binary classification.
    """
    
    input_shape = (7781, 1)  # Assuming each element in X_train is a 1D array
    
    # Input layer
    inputs = Input(shape=input_shape)

    x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    for j, k in enumerate(n_node_CNN): 
        x = Conv1D(k, kernel_size=5, activation='relu')(x)
        if j<(len(n_node_CNN)): 
            x = MaxPooling1D(pool_size=2)(x)

    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully Connected Layer
    for i, n in enumerate(n_node_FC): 
        x = Dense(n, activation='relu')(x)

    
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    mod1 = Model(inputs=inputs, outputs=outputs)
    return mod1