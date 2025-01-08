import numpy as np
import pandas as pd
import warnings
import matplotlib as plt

def get_fit(points):
    z_points = points.loc[:,"z"]
    x_points = points.loc[:,"x"]
    y_points = points.loc[:,"y"]
    t_points = points.loc[:,"t"]
    x_poly = plot_points_and_fit_line(t_points,x_points,1)
    y_poly = plot_points_and_fit_line(t_points,y_points,1)
    z_poly = plot_points_and_fit_line(t_points,z_points,2)


    zeros = find_zero_of_quadratic_fit(z_points,t_points)
    assert (len(zeros) == 2)
    first,last = min(zeros[0],zeros[1]),max(zeros[0],zeros[1])
    first_point, last_point = (x_poly(first),y_poly(first),0),(x_poly(last),y_poly(last),0)
    return first_point,last_point

def plot_points_and_fit_line(x, t, deg):
    """
    Plots points over time and fits a line using numpy's polyfit function.

    Parameters:
    x (list): List of data points.
    t (list): List of time points corresponding to the data points.

    Returns:
    polynomial
    """
    # Ensure x and t are numpy arrays for numerical operations
    x = np.array(x)
    t = np.array(t)

    # Fit a line to the data points
    coefficients = np.polyfit(t, x, deg)
    polynomial = np.poly1d(coefficients)

    # Generate values for the fitted line
    x_fit = polynomial(t)

    # Plot the original data points
    plt.scatter(t, x, color='blue', label='Data Points')

    # Plot the fitted line
    plt.plot(t, x_fit, color='red', label='Fitted Line')

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Data Points')
    plt.title('Data Points and Fitted Line')
    plt.legend()

    # Show the plot
    plt.show()
    return x_fit

def find_zero_of_quadratic_fit(y, t):
    """
    Finds the time t at which a quadratic fit of y over t equals zero.

    Parameters:
    y (list): A list of data points.
    t (list): A list of time points.

    Returns:
    float or None: The time t where the quadratic fit equals zero, if such a point exists.
    """
    # Perform a quadratic fit (degree 2) using numpy's polyfit
    coefficients = np.polyfit(t, y, 2)

    # Find the roots of the quadratic equation
    roots = np.roots(coefficients)

    # Filter out complex roots and return real roots
    real_roots = [root.real for root in roots if np.isreal(root)]

    # Return the first real root if it exists, otherwise return None
    return real_roots[0] if real_roots else None

