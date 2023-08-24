"""
Code to fit a 2D pixel array to a 2D Gaussian.

Tony Fu, June 17, 2022
"""
from typing import Tuple, NamedTuple

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


__all__ = ['fit_2d_gaussian', 'calc_f_explained_var', 'GaussianParameters', 'calculate_2d_gaussian']


class GaussianParameters(NamedTuple):
    """
    amplitude: int or float
        The amplitude of the Gaussian, i.e., the max of the Gaussian if the
        offset is zero. Unitless.
    mu_x & mu_y: int or float
        The center coordiates of the Gaussian, in unit pixels.
    sigma_1 & sigma_2: int or float
        The std. dev. of the two orthogonal axis in unit pixels. sigma_1
        should be the horizontal axis if theta = 0 degree.
    theta: int or float
        The angle of the sigma_1 away from the positive x-axis, measured
        counterclockwise in unit degrees.
    offset: int or float
        The offset of the Gaussian. Unitless.
    """
    amplitude: float
    mu_x: float
    mu_y: float
    sigma_1: float
    sigma_2: float
    theta: float
    offset: float


def calculate_2d_gaussian(xycoord: Tuple[np.ndarray, np.ndarray],
                           params: GaussianParameters) -> np.ndarray:
    """
    The model function of a 2D Gaussian. Intended to be the first input
    argument for scipy.optimize.curve_fit(f, xdata, ydata).

    Parameters
    ----------
    xycoord: a two-tuple of 2D numpy arrays = (x, y)
        The x and y coordinates, in unit pixels. x and y should be 2D array,
        i.e., the result of meshgrid(x, y).
    params: GaussianParameters
        NamedTuple containing parameters of the Gaussian including
        amplitude, mu_x, mu_y, sigma_1, sigma_2, theta, and offset.

    Returns
    -------
    z: 2D numpy array
        The height of the 2D Gaussian given coordinates (x, y). This has to be
        a 1D array as per the official documentation of
        scipy.optimize.curve_fit().
    """
    x, y = xycoord
    mu_x, mu_y, theta = float(params.mu_x), float(params.mu_y), params.theta * np.pi / 180
    sigma_1, sigma_2 = params.sigma_1, params.sigma_2
    a = (np.cos(theta)**2)/(2*sigma_1**2) + (np.sin(theta)**2)/(2*sigma_2**2)
    b = -(np.sin(2*theta))/(4*sigma_1**2) + (np.sin(2*theta))/(4*sigma_2**2)
    c = (np.sin(theta)**2)/(2*sigma_1**2) + (np.cos(theta)**2)/(2*sigma_2**2)
    z = params.offset + params.amplitude*np.exp(-(a*((x-mu_x)**2)
                                                  + 2*b*(x-mu_x)*(y-mu_y)
                                                  + c*((y-mu_y)**2)))
    return z.flatten()


def _prepare_initial_guess(image: np.ndarray) -> GaussianParameters:
    amp_guess = image.max() - image.min()
    muy_guess, mux_guess = np.unravel_index(np.argmax(image), image.shape)
    sigma_1_guess = image.shape[0] / 4
    sigma_2_guess = image.shape[0] / 5
    theta_guess = 90
    offset_guess = image.mean()

    return GaussianParameters(amp_guess, mux_guess, muy_guess, sigma_1_guess, sigma_2_guess, theta_guess, offset_guess)


def fit_2d_gaussian(image: np.ndarray,
                    initial_guess: GaussianParameters = None,
                    plot: bool = True,
                    show: bool = False,
                    cmap=plt.cm.jet) -> Tuple[GaussianParameters, np.ndarray]:
    """
    Fit a 2D gaussian to an input image.

    Parameters
    ----------
    image: 2D numpy array
        The image to be fit. The pixel [0, 0] is the top-left corner.
    intial_guess: GaussianParameters (NamedTuple)
        Initial guesses for the parameters (see Returns).
    plot: bool
        Whether to plot the result isocline or not.
    plot: bool
        Whether to show the result plot.

    Returns
    -------
    param_estimate: GaussianParameters (NamedTuple)
    param_sem: numpy.array
        The standard errors of the parameter estimates listed above.
    """
    # Create x and y indices.
    y_size, x_size = image.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)
    
    # Wrapper function to unpack parameters
    def wrapper(xycoord, amplitude, mu_x, mu_y, sigma_1, sigma_2, theta, offset):
        params = GaussianParameters(amplitude, mu_x, mu_y, sigma_1, sigma_2, theta, offset)
        return calculate_2d_gaussian(xycoord, params)

    # Initialize initial guess.
    if initial_guess is None:
        initial_guess = _prepare_initial_guess(image)

    # Fit the 2D Gaussian.
    try:
        param_estimate, params_covar = opt.curve_fit(wrapper,
                                                    (x, y),
                                                    image.flatten(),
                                                    p0=initial_guess,
                                                    method='lm',
                                                    check_finite=True)
        param_estimate = GaussianParameters(*param_estimate)
        param_sem = GaussianParameters(*np.sqrt(np.diag(params_covar)))
    except:
        param_estimate = GaussianParameters(*([-1] * len(GaussianParameters._fields)))
        param_sem = GaussianParameters(*([-999] * len(GaussianParameters._fields)))
        print("bad fit")

    # Plot the original image with the fitted curves.
    if plot:
        image_fitted = calculate_2d_gaussian((x, y), param_estimate)
        plt.imshow(image, cmap=cmap)
        plt.contour(x, y, image_fitted.reshape(y_size, x_size), 9, colors='w')

    if show:
        plt.show()

    return param_estimate, param_sem


def calc_f_explained_var(sum_map: np.ndarray, params: GaussianParameters) -> float:
    """
    Calculates the fraction of variance explained by the fit with the formula:
        exp_var = 1 - var(sum_map - fit_map)/var(sum_map)

    Parameters
    ----------
    sum_map : numpy.ndarray
        The map constructed by summing bars.
    params : GaussianParameters
        Parameters of 2D elliptical Gaussian fit.

    Returns
    -------
    exp_var : float
        The fraction of explained variance.
    """
    # Reconstruct map with fit parameters.
    x_size = sum_map.shape[1]
    y_size = sum_map.shape[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)
    fit_map = calculate_2d_gaussian((x, y), params)

    # Calculate variances
    residual_var = np.var(fit_map - sum_map.flatten())
    gt_var = np.var(sum_map.flatten())
    return 1 - (residual_var / gt_var)
