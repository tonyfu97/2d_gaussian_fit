import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from gaussian_2d_fitting import (fit_2d_gaussian,
                                 calc_f_explained_var,
                                 calculate_2d_gaussian,
                                 GaussianParameters,)


###############################################################################

# Example 1. Fit a 2D Gaussian to a single Gaussian.

# Define the size of the data.
x_size, y_size = 200, 250

# Create x and y indices using meshgrid.
x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))

# Define the Gaussian parameters using the NamedTuple.
gaussian_params = GaussianParameters(amplitude=3,
                                     mu_x=80,
                                     mu_y=180,
                                     sigma_1=40,
                                     sigma_2=70,
                                     theta=45,
                                     offset=10)

# Create data using the 2D Gaussian function and add some noise.
data = calculate_2d_gaussian((x, y), gaussian_params).reshape(y_size, x_size)
data_noisy = data + 0.2 * np.random.normal(size=data.shape)

# Fit the 2D Gaussian to the noisy data and plot the result.
param_estimate, param_sem = fit_2d_gaussian(data_noisy, plot=True, show=True)

# Print the estimated parameters and their standard errors.
print("Estimated Parameters:", param_estimate)
print("Standard Errors:", param_sem)

# Print the explained variance.
print("Explained Variance:", calc_f_explained_var(data_noisy, param_estimate))


###############################################################################

# Example 2. Fit a 2D Gaussian to a sum of two Gaussians.

# Define the parameters for two different Gaussians.
gaussian_params1 = GaussianParameters(amplitude=2,
                                      mu_x=50,
                                      mu_y=120,
                                      sigma_1=30,
                                      sigma_2=50,
                                      theta=30,
                                      offset=5)
gaussian_params2 = GaussianParameters(amplitude=4,
                                      mu_x=150,
                                      mu_y=180,
                                      sigma_1=20,
                                      sigma_2=40,
                                      theta=60,
                                      offset=5)

# Create data for the two Gaussians and add them together.
data1 = calculate_2d_gaussian((x, y), gaussian_params1).reshape(y_size, x_size)
data2 = calculate_2d_gaussian((x, y), gaussian_params2).reshape(y_size, x_size)
data_combined = data1 + data2 + 0.2 * np.random.normal(size=data1.shape)

# Fit the 2D Gaussian to the combined data.
param_estimate_combined, param_sem_combined = fit_2d_gaussian(data_combined, plot=True, show=True)

print("Estimated Parameters for Combined Data:", param_estimate_combined)
print("Standard Errors for Combined Data:", param_sem_combined)

# Print the explained variance.
print("Explained Variance for Combined Data:", calc_f_explained_var(data_combined, param_estimate_combined))
plt.close()


###############################################################################

# Example 3. Use multiprocessing to fit multiple images and save the fit in a multipage PDF file.

def fit_image(image):
    return fit_2d_gaussian(image, plot=False)

def create_noisy_data(base_params, noise_level=0.2):
    noisy_params = GaussianParameters(*[param + noise_level * np.random.normal() for param in base_params])
    data = calculate_2d_gaussian((x, y), noisy_params).reshape(y_size, x_size)
    return data + 0.2 * np.random.normal(size=data.shape)

if __name__ == '__main__':
    base_gaussian_params = GaussianParameters(amplitude=3,
                                              mu_x=80,
                                              mu_y=180,
                                              sigma_1=40,
                                              sigma_2=70,
                                              theta=45,
                                              offset=10)

    images = [create_noisy_data(base_gaussian_params) for _ in range(10)]

    with Pool() as pool:
        results = list(tqdm(pool.imap(fit_image, images), total=len(images)))

    with PdfPages('./results/fit_results.pdf') as pdf, open('./results/fit_results.txt', 'w') as txt_file:
        for i, (image, (params, sem)) in enumerate(zip(images, results)):
            plt.imshow(image, cmap=plt.cm.jet)
            x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))
            image_fitted = calculate_2d_gaussian((x, y), params)
            plt.contour(x, y, image_fitted.reshape(y_size, x_size), 9, colors='w')

            # Add title with estimated parameters and explained variance
            fxvar = calc_f_explained_var(image, params)
            title = f"Image {i}: FxVar: {fxvar:.4f}"
            plt.title(title)

            pdf.savefig()
            plt.close()

            # Write results to a text file
            txt_file.write(f"Image {i}:\n")
            txt_file.write(f"Estimated Parameters: {params}\n")
            txt_file.write(f"Standard Errors: {sem}\n")
            txt_file.write(f"Explained Variance: {fxvar}\n")
            txt_file.write(os.linesep)
