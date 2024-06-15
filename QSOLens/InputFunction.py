import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Activation, Dense, Dropout
from NetworkArchitecture import network_classifier
from NetworkArchitecture import network_z_finder


def predict(array, Z_find = None):
    """
    Predicts the output using the network architecture defined in the `network` function.
    
    Parameters:
    array (np.ndarray): Input data array to be used for prediction.
    
    Returns:
    np.ndarray: Predicted output from the model.
    """
    if Z_find == False:
        mod1 = network_classifier(array)
        mod1.load_weights('new_model_weights_3.h5')
        predict_lens = mod1.predict(array)
        return predict_lens
    else:
        mod2 = network_z_finder(array)
        mod2.load_weights('new_model_weights_3.h5')
        predict_redshift = mod2.predict(array)
        return predict_redshift



