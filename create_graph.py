import numpy as np
import pandas as pd
import warnings

import tqdm
from sklearn.cluster import KMeans as KMeans_
import matplotlib.pyplot as plt



X = "x"
Y = "y"
Z = "z"

TIME = "time"
def get_fit(points):
    x_points,y_points,z_points,t_points = extract_arrays(points)
    x_poly = np.poly1d(plot_points_and_fit_line(t_points,x_points,1))
    y_poly = np.poly1d(plot_points_and_fit_line(t_points,y_points,1))
    z_poly = plot_points_and_fit_line(t_points,z_points,2)


    zeros = find_zero_of_quadratic_fit(z_points,t_points)
    if (zeros != None and len(zeros) == 2):
        first,last = min(zeros[0],zeros[1]),max(zeros[0],zeros[1])
        first_point, last_point = (x_poly(first),y_poly(first),0),(x_poly(last),y_poly(last),0)
        return first_point,last_point
    return (0,0,0),(0,0,0)

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

    # # Plot the original data points
    # plt.scatter(t, x, color='blue', label='Data Points')
    #
    # # Plot the fitted line
    # plt.plot(t, x_fit, color='red', label='Fitted Line')

    # # Add labels and legend
    # plt.xlabel('Time')
    # plt.ylabel('Data Points')
    # plt.title('Data Points and Fitted Line')
    # plt.legend()

    # Show the plot
    # plt.show()
    return coefficients

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
    return real_roots if real_roots else None


def run(n_rockets):
    points = []
    for i in tqdm.tqdm(range(len(n_rockets))):
        point = get_fit(n_rockets[i][1])
        points.append(point[0][0:2])
    testList2 = np.array([(elem1, elem2) for elem1, elem2 in points])

    x_points,y_points = [x for x,_ in testList2], [y for _,y in testList2]
    plt.scatter(x_points,y_points,color='red', label='Fitted Line')
    plt.show()

    kmeans = KMeans_(n_clusters=50,random_state=42)
    clusters = kmeans.fit(points)
    centres = kmeans.cluster_centers_

    newx_points, newy_points = [x for x, _ in centres], [y for _, y in centres]
    plt.scatter(newx_points,newy_points, color='blue', label='centers')
    plt.show()

def extract_arrays(df):
    """
    Extracts X, Y, T, and Z arrays from a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): Column name for X array.
        y_col (str): Column name for Y array.
        t_col (str): Column name for T array.
        z_col (str): Column name for Z array.

    Returns:
        tuple: A tuple containing X, Y, T, and Z arrays (as numpy arrays).
    """
    x_points = df[X].to_numpy()
    y_points = df[Y].to_numpy()
    t_points = df[TIME].to_numpy()
    z_points = df[Z].to_numpy()
    return x_points,y_points,z_points,t_points