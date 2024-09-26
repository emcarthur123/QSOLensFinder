import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class QSOFluxProcessor:
    def __init__(self):
        # Initialize the classifier and redshift models
        self.classifier_model = self.build_classifier_model()
        self.redshift_model = self.build_redshift_model()
        self.waves = np.load('wave.npy', allow_pickle=True)  # Load wave data from file

    def build_classifier_model(self, input_layer=50, n_node_CNN=[50, 50, 100, 100, 100], n_node_FC=[30, 25]):
        input_shape = (7781, 1)
        inputs = Input(shape=input_shape)
        x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        for k in n_node_CNN:
            x = Conv1D(k, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        for n in n_node_FC:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights('Weights/Phase1.h5')
        return model

    def build_redshift_model(self, input_layer=50, n_node_CNN=[50, 50, 100, 100, 100], n_node_FC=[30, 25]):
        input_shape = (7781, 1)
        inputs = Input(shape=input_shape)
        x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        for k in n_node_CNN:
            x = Conv1D(k, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        for n in n_node_FC:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights('Weights/Redshift.h5')
        return model

    def preprocess_data(self, array):
        # Standardize and normalize the input fluxes
        standardized = np.array([((flux - np.mean(flux)) / np.std(flux)) for flux in array])
        stn = np.array([x / np.max(np.abs(x)) for x in standardized])
        stn = np.expand_dims(stn, axis=-1)
        return stn

    def classify(self, fluxes):
        # Classify the fluxes
        preprocessed = self.preprocess_data(fluxes)
        return self.classifier_model.predict(preprocessed)

    def predict_redshift(self, fluxes):
        # Predict the redshift using the preprocessed fluxes
        preprocessed = self.preprocess_data(fluxes)
        return self.redshift_model.predict(preprocessed)

    # Gaussian function definition
    def gaussian(self, x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    # Model function: Gaussian + continuum
    def model(self, x_value, amp, mean, sigma, cont):
        return self.gaussian(x_value, amp, mean, sigma) + cont

    # Chi-squared function including noise
    def chi_squared(self, params, wave, flux):
        amp, mean, sigma, cont = params
        model_flux = self.model(wave, amp, mean, sigma, cont)
        chi_sq = np.sum(((flux - model_flux)) ** 2)
        return chi_sq

    def fit_gaussian(self, flux, wave, redshift):
        try:
            redshift_low = redshift - 0.03
            redshift_high = redshift + 0.03
            mean_wavelength = 3727  # OII doublet center
            wavelength_low = mean_wavelength * (1 + redshift_low)
            wavelength_high = mean_wavelength * (1 + redshift_high)

            # Create mask for the truncated range
            mask = (wave >= wavelength_low) & (wave <= wavelength_high)
            truncated_wave = wave[mask]
            truncated_flux = flux[mask]

            # Check if there is any valid data to fit
            if len(truncated_wave) == 0 or len(truncated_flux) == 0:
                raise ValueError("Empty truncated flux or wavelength arrays.")

            # Find the peak of the flux for initial guess
            peak_index = np.argmax(truncated_flux)
            initial_mean = truncated_wave[peak_index]

            # Initial guesses for the parameters
            initial_amp = np.max(truncated_flux)
            initial_sigma = 4
            initial_cont = np.median(truncated_flux)
            initial_params = [initial_amp, initial_mean, initial_sigma, initial_cont]

            # Fit the model using scipy's minimize
            result = minimize(self.chi_squared, initial_params, args=(truncated_wave, truncated_flux))

            # Check if the optimization was successful
            if not result.success:
                raise ValueError(f"Optimization failed: {result.message}")

            # Optimized parameters
            opt_params = result.x
            fitted_redshift = (opt_params[1] - mean_wavelength) / mean_wavelength

            # Return the optimized parameters and redshift
            return fitted_redshift, opt_params[0], opt_params[1], opt_params[2], opt_params[3], result.fun, 0  # flag = 0 (success)

        except Exception as e:
            # If any error occurs, return np.nan values and set the flag to 1
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.inf, 1  # flag = 1 (error)

    def fit_and_plot_gaussian(self, flux, redshift):
        """
        Fit a Gaussian model to the provided flux and visualize the result.
        
        Parameters:
        flux (numpy.ndarray): The flux data to fit.
        redshift (float): The redshift value to use in the fitting.
        """
        wave = self.waves  # Ensure the corresponding wavelength data
        fitted_redshift, amp, mean, sigma, cont, best_chi_squared, flag = self.fit_gaussian(flux, wave, redshift)
        
        redshift_low_plot = redshift - 0.03
        redshift_high_plot = redshift + 0.03
        mean_wavelength = 3727  # OII doublet center
        wavelength_low_plot = mean_wavelength * (1 + redshift_low_plot)
        wavelength_high_plot = mean_wavelength * (1 + redshift_high_plot)

        # Create mask for the truncated range
        mask = (wave >= wavelength_low_plot) & (wave <= wavelength_high_plot)
        truncated_wave = wave[mask]
        truncated_flux = flux[mask]

        
        # Plot the original flux and the fitted model
        plt.figure(figsize=(10, 6))
        plt.plot(truncated_wave, truncated_flux, label='Original Flux', color='blue')
        
        # If fitting was successful, plot the fitted model
        if flag == 0:
            # Generate fitted model
            fitted_flux = self.model(wave, amp, mean, sigma, cont)
            plt.plot(wave[mask], fitted_flux[mask], label='Fitted Gaussian Model', color='red')
            plt.title('Gaussian Fit to Flux Data')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            plt.title('Gaussian Fitting Failed')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.text(0.5, 0.5, 'Fitting failed. Please check the data.', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=plt.gca().transAxes, fontsize=12)
            plt.grid()
            plt.show()

    def process(self, fluxes, names=None, noise=None):
        # Classify fluxes
        predictions_class = self.classify(fluxes)
        predictions_z = self.predict_redshift(fluxes)

        results = []

        for i in range(len(fluxes)):
            original_flux = fluxes[i]
            wave = self.waves  # Ensure the corresponding wavelength data
            predicted_redshift = predictions_z[i][0]  # Use the redshift predicted by the redshift network

            # Perform Gaussian fitting using the raw flux and wavelength
            fitted_redshift, amp, mean, sigma, cont, best_chi_squared, flag = self.fit_gaussian(original_flux, wave, predicted_redshift)

            # Determine classification based on the score
            score = predictions_class[i][0]
            classification = 'Lens' if score > 0.5 else 'QSO'

            # Handle names
            if names is not None and i < len(names):
                name = names[i]
            else:
                name = i + 1  # Incremental name if not provided

            # Calculate SNR for CNN prediction
            if noise is not None:
                snr_cnn = (np.sum(original_flux - cont) / np.sum(noise)) if np.sum(noise) != 0 else 0
            else:
                snr_cnn = 0

            # Calculate SNR for Gaussian prediction
            if noise is not None:
                snr_gaussian = (np.sum(fitted_redshift - cont) / np.sum(noise)) if np.sum(noise) != 0 else 0
            else:
                snr_gaussian = 0

            # Append results including classification, redshift, and fitting results
            results.append([name, classification, predicted_redshift, fitted_redshift, best_chi_squared, amp, mean, sigma, cont, flag, snr_cnn, snr_gaussian])

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results, columns=['Name', 'Classification', 'Predicted Redshift', 'Fitted Redshift', 'Best Chi-Squared', 'Amplitude', 'Mean', 'Sigma', 'Continuum', 'Flag', 'SNR CNN', 'SNR Gaussian'])

        return results_df
