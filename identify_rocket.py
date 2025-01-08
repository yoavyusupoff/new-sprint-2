import numpy as np
import pandas as pd
import warnings

from csv_to_data import *
from create_graph import *
from rocket import *

def get_fit(x_points, y_points, z_points, t_points):
    x_poly = np.poly1d(plot_points_and_fit_line(t_points,x_points,1))
    y_poly = np.poly1d(plot_points_and_fit_line(t_points,y_points,1))
    z_poly = plot_points_and_fit_line(t_points,z_points,2)
    zeros = find_zero_of_quadratic_fit(z_points,t_points)
    if (zeros != None and len(zeros) == 2):
        first,last = min(zeros[0],zeros[1]),max(zeros[0],zeros[1])
        first_point, last_point = np.array([x_poly(first),y_poly(first),0]),np.array([x_poly(last),y_poly(last),0])
        rocket_range = np.linalg.norm(last_point - first_point)
        return rocket_range

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

def find_start_angle_of_quadratic_fit(z_poly_coefficients):
    # Find the roots (x-intercepts)
    roots = np.sort(np.roots(z_poly_coefficients))
    try:
        x_cross = roots[np.isreal(roots)].real[0]  # Take the first real root
    except IndexError:
        x_cross = 0
    # Compute the derivative of the polynomial
    derivative_coefficients = np.polyder(z_poly_coefficients)

    # Evaluate the slope at the crossing point
    slope = np.polyval(derivative_coefficients, x_cross)

    # Calculate the angle in radians and degrees
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def extract_arrays(trajectory: np.ndarray):
    t_points = trajectory[:, 0]
    x_points = trajectory[:, 1]
    y_points = trajectory[:, 2]
    z_points = trajectory[:, 3]
    return x_points,y_points,z_points,t_points

def identify_rocket_type_of_launcher_by_range(rocket_trajectories:np.ndarray) -> None:
    rockets_data = []
    for rocket_trajectory in rocket_trajectories:
        x_points, y_points, z_points, t_points = extract_arrays(rocket_trajectory)
        z_poly = plot_points_and_fit_line(z_points, t_points, 2)

        launch_angle = find_start_angle_of_quadratic_fit(z_poly)
        range = get_fit(x_points, y_points, z_points, t_points)
        rockets_data.append((launch_angle, range))
    return find_best_rocket_by_range(np.array(rockets_data))

def get_launcher_trajectories(keys):
    d = ramot_main()
    rocket_trajectories = [d[key] for key in keys]
    return rocket_trajectories

def calculate_max_velocity(t_points, x_points, y_points, z_points):
    """
    Calculate the maximal velocity of a rocket given its trajectory.

    Parameters:
        t_points (list or np.ndarray): Time points.
        x_points (list or np.ndarray): X-coordinates of the trajectory.
        y_points (list or np.ndarray): Y-coordinates of the trajectory.
        z_points (list or np.ndarray): Z-coordinates of the trajectory.

    Returns:
        float: Maximal velocity magnitude.
    """
    # Convert to NumPy arrays
    t_points = np.array(t_points)
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    z_points = np.array(z_points)

    i = 1
    for _ in range(10):
        t_points = delete_noise(t_points, i)
        x_points = delete_noise(x_points, i)
        y_points = delete_noise(y_points, i)
        z_points = delete_noise(z_points, i)

    # Calculate differences in time and positions
    dt = np.diff(t_points)
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    dz = np.diff(z_points)

    # Avoid division by zero for small time differences
    if np.any(dt <= 0):
        raise ValueError("Time points must be strictly increasing.")

    # Calculate velocity components
    v_x = dx / dt
    v_y = dy / dt
    v_z = dz / dt



    # Calculate velocity magnitudes
    velocities = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    max_velocity = np.max(velocities[dz > 0])
    velocities = np.append(velocities, velocities[-1])
    for value in velocities:
        print(value)
    plt.plot(t_points, velocities)
    plt.show()
    # Find and return the maximum velocity
    return max_velocity

def delete_noise(arr, i):
    shifted_forward_arr = np.empty_like(arr)
    shifted_forward_arr[i:] = arr[:-i]
    shifted_forward_arr[0:i] = 0

    shifted_backward_arr = np.empty_like(arr)
    shifted_backward_arr[0:-i] = arr[i:]
    shifted_backward_arr[-i:] = arr[-1]

    return (shifted_forward_arr + shifted_backward_arr) / 2



def identify_rocket_type_of_launcher_by_burst_velocity(rocket_trajectories:np.ndarray) -> None:
    rockets_data = []
    for rocket_trajectory in rocket_trajectories:
        x_points, y_points, z_points, t_points = extract_arrays(rocket_trajectory)
        return calculate_max_velocity(t_points, x_points, y_points, z_points)

def get_eylon_data(gus_result):
    eylon_data = []
    # dict = ramot_main()
    # arr = data_to_graph(dict)
    for value in gus_result:
        keys = value[0]
        position = value[1]
        rocket_type, accuracy = identify_rocket_type_of_launcher_by_range(get_launcher_trajectories(keys))
        eylon_data.append((position, rocket_type, accuracy))
        print(eylon_data)
    return eylon_data




