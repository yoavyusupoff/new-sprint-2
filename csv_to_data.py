import os
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as KMeans_

# import create_graph
from utils import sphere_to_xyz

# ___________________________________________________________________________


FOLDER_PATH = "./data/With ID/Target bank data"
FILENAME_SUFFIX = "_with_ID.csv"
TIME_OFFSET = 1736300000

ID = "ID"
RADAR_NAME = "radar_name"
PHI = "phi"

X = "x"
Y = "y"
Z = "z"

TIME = "time"
RANGE = "range"
ELEVATION = "elevation"
AZIMUTH = "azimuth"
RAD_VEL = "radial_velocity"

RANGE_UC = "range_uncertainty"
ELEVATION_UC = "elevation_uncertainty"
AZIMUTH_UC = "azimuth_uncertainty"
RAD_VEL_UC = "radial_velocity_uncertainty"


# ___________________________________________________________________________


def get_all_filenames(folder_path: str) -> List[str]:
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(FILENAME_SUFFIX):
            file_path = os.path.join(folder_path, filename)
            filenames.append(file_path)

    return filenames


# ___________________________________________________________________________


def modify_sub_table(sub_table: pd.DataFrame, radar_name: str) -> pd.DataFrame:
    # drop the ID column and reset the table indexes
    result = sub_table.drop(sub_table.columns[-1], axis=1)
    result = result.reset_index(drop=True)
    result.index += 1

    # add a column of the radar name
    result.insert(0, RADAR_NAME, radar_name)

    # remove the time offset to work with more logical time
    result[TIME] -= TIME_OFFSET

    # convert the range from meters to kilometers
    result[RANGE] = result[RANGE] / 1000

    # replace elevation column with phi column (90 - alpha)
    result.insert(result.columns.get_loc(ELEVATION), PHI, 90 - result[ELEVATION])
    result.drop(columns=[ELEVATION], inplace=True)

    return result


def create_id_to_data_map(radar_filename: str, folder_path: str) -> Dict[int, pd.DataFrame]:
    # read the csv file of the radar data and drop the unnamed first column
    data_frame = pd.read_csv(radar_filename)
    data_frame.drop(data_frame.columns[0], axis=1, inplace=True)

    # slice to get the radar name
    name_start = len(folder_path) + 1
    name_end = radar_filename.find(FILENAME_SUFFIX)
    radar_name = radar_filename[name_start:name_end]

    # return list mapping of ID |-> data table
    return {int(key): modify_sub_table(sub_table, radar_name)
            for (key, sub_table) in data_frame.groupby(ID)}


# ___________________________________________________________________________


def merge_id_to_data_maps(folder_path: str) -> Dict[int, pd.DataFrame]:
    result_map = dict()

    for radar_filename in get_all_filenames(folder_path):
        # get the mapping of the current radar
        id_to_data_map = create_id_to_data_map(radar_filename, folder_path)

        for rocket_id, data_table in id_to_data_map.items():
            # if the data is of a previously-visited radar, append to the existing data
            if rocket_id in result_map:
                combined = pd.concat([result_map.get(rocket_id), data_table], ignore_index=True)
                result_map[rocket_id] = combined
            else:
                result_map[rocket_id] = data_table

            # sort the rows by the time
            result_map.get(rocket_id).sort_values(by=TIME, ascending=True, ignore_index=True,
                                                  inplace=True)

    return result_map


# ___________________________________________________________________________


def convert_dict_to_list_of_tuples(d: dict):
    return list(sorted(d.items()))


# ___________________________________________________________________________


def convert_sphere_table_to_cartesian(table: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()

    # copy the time column
    # result[RADAR_NAME] = table[RADAR_NAME]
    result[TIME] = table[TIME]

    x_list, y_list, z_list = [], [], []

    for index, row in table.iterrows():
        r, phi, theta = row[RANGE], row[PHI], row[AZIMUTH]
        x, y, z = sphere_to_xyz(row[RADAR_NAME], (r, phi, theta))

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    # add the X, Y, Z columns to the table
    result[X] = pd.DataFrame(x_list)
    result[Y] = pd.DataFrame(y_list)
    result[Z] = pd.DataFrame(z_list)

    return result


def create_id_to_cartesian_map(id_to_data_map: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    return {rocket_id: convert_sphere_table_to_cartesian(data_table)
            for rocket_id, data_table in tqdm.tqdm(id_to_data_map.items(), desc="Converting to Cartesian")}


# ___________________________________________________________________________


def get_numpy_result(folder_path: str) -> Dict[int, np.ndarray]:
    merged = merge_id_to_data_maps(folder_path)
    merged_in_cartesian = create_id_to_cartesian_map(merged)

    result = dict()
    for key, id_map in convert_dict_to_list_of_tuples(merged_in_cartesian):
        id_map.drop(columns=[RADAR_NAME], inplace=True)
        result[key] = id_map.to_numpy()

    return result


def load_cartesian_map():
    with open(PKL, "rb") as file:
        cartesian_map = pickle.load(file)
    return cartesian_map


def get_launch_points_clustered(dic: Dict):
    all_launches = []
    launches_id = []
    for rocket_id, mat in dic.items():
        mat = dic[rocket_id]
        t = mat[TIME]
        x = mat[X]
        y = mat[Y]
        z = mat[Z]
        if get_launch_point(x, y, z, t) is None:
            continue
        x_l, y_l = get_launch_point(x, y, z, t)
        launches_id.append(rocket_id)
        all_launches.append((x_l, y_l))

    all_launches = [x for x in all_launches if -50 <= x[0] <= 10 and -70 <= x[1] <= 10]
    launches_id = [launches_id[x] for x in range(len(all_launches)) if -50 <= all_launches[x][0] <= 10 and -70 <= all_launches[x][1] <= 10]
    # Plot all_launches
    x_coords = [point[0] for point in all_launches]
    y_coords = [point[1] for point in all_launches]

    # Plot the points
    # plt.scatter(x_coords, y_coords, color='blue', label='Points')
    # # plt.plot(x_coords, y_coords, linestyle='--', color='orange', label='Line')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('2D Points Plot')
    # plt.legend()
    # plt.grid()
    # plt.show()

    kmeans = kmeans_with_min_cluster_size(data=all_launches, n_clusters=50, min_size=10, max_iter=300, random_state=None)

    centres = kmeans[1]
    labels = kmeans[0]
    newx_points, newy_points = [x for x, _ in centres], [y for _, y in centres]
    # plt.scatter(newx_points,newy_points, color='blue', label='centers')
    # plt.show()

    print([([launches_id[j] for j in range(len(all_launches)) if labels[j] == i], centres[i]) for i in range(50)])
    return [([launches_id[j] for j in range(len(all_launches)) if labels[j] == i], centres[i]) for i in range(50)]


def kmeans_with_min_cluster_size(data, n_clusters, min_size, max_iter=1000, random_state=None):
    """
    Perform KMeans clustering ensuring each cluster has at least `min_size` items.

    Parameters:
        data (ndarray): The dataset (n_samples, n_features).
        n_clusters (int): Number of clusters.
        min_size (int): Minimum number of items per cluster.
        max_iter (int): Maximum number of iterations for KMeans.
        random_state (int): Random seed for reproducibility.

    Returns:
        labels (ndarray): Cluster assignments for each data point.
        centers (ndarray): Coordinates of cluster centers.
    """
    kmeans = KMeans_(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Recheck cluster sizes and adjust if necessary
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        small_clusters = [k for k, size in cluster_sizes.items() if size < min_size]

        if not small_clusters:
            break  # All clusters meet the minimum size

        # Find the largest cluster to redistribute points
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        largest_cluster_indices = np.where(labels == largest_cluster)[0]

        # Reassign points from the largest cluster to small clusters
        for small_cluster in small_clusters:
            deficit = min_size - cluster_sizes[small_cluster]
            if len(largest_cluster_indices) < deficit:
                deficit = len(largest_cluster_indices)

            reassign_indices = largest_cluster_indices[:deficit]
            largest_cluster_indices = largest_cluster_indices[deficit:]
            labels[reassign_indices] = small_cluster

    return labels, centers

def get_launch_point(x,y,z,t):
    return get_launch_finish_points(x,y,z,t)[0]

def get_hit_point(x,y,z,t):
    return get_launch_finish_points(x,y,z,t)[1]
def get_launch_finish_points(x, y, z, t):
    first_part_len = 2*len(x) // 3

    x_start = x[:first_part_len]
    y_start = y[:first_part_len]
    z_start = z[:first_part_len]
    t_start = t[:first_part_len]

    x_hitt = x[first_part_len:]
    y_hitt = y[first_part_len::]
    z_hitt = z[first_part_len::]
    t_hitt = t[first_part_len::]
    coefficients = np.polyfit(t, z, 2)  # Returns [a, b, c] for at^2 + bt + c

    a, b, c = coefficients

    # Target value to find hit time
    z_hit = 0  # Adjust this value as needed

    # Solve the quadratic equation: at^2 + bt + (c - z_hit) = 0
    c_prime = c - z_hit
    discriminant = b ** 2 - 4 * a * c_prime

    if discriminant <= 0:
        return None
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    fit_deg = 2
    x_0 = polynomial_fit_and_predict(t_start, x_start, t1, deg=fit_deg)
    y_0 = polynomial_fit_and_predict(t_start, y_start, t1, deg=fit_deg)

    x_1,y_1 = polynomial_fit_and_predict(t_hitt, x_hitt, t2, deg=fit_deg), polynomial_fit_and_predict(t_hitt, y_hitt, t2, deg=fit_deg)
    # # Plot polyfit of deg fit_deg of x and y as a function of t
    # plt.scatter(t, np.polyval(np.polyfit(t, x, fit_deg), t), color='black', label='x')
    # plt.scatter(t, np.polyval(np.polyfit(t, y, fit_deg), t), color='black', label='y')
    # # Plot x, y, z as function of t and mark 0 at x_0, y_0
    # plt.scatter(t, x, color='blue', label='x')
    # plt.scatter(t, y, color='red', label='y')
    # plt.scatter(t, z, color='yellow', label='z')
    # # Add x mark at t1, x0
    # plt.scatter(t1, x_0, color='blue', marker='x', label='x_0')
    # plt.scatter(t1, y_0, color='red', marker='x', label='y_0')
    # plt.show()

    return [(x_0, y_0),(x_1,y_1)]
    # plt.show()


def get_landing_point(x, y, z, t):
    first_part_len = len(x) // 3
    x = x[first_part_len:]
    y = y[first_part_len:]
    z = z[first_part_len:]
    t = t[first_part_len:]
    coefficients = np.polyfit(t, z, 2)  # Returns [a, b, c] for at^2 + bt + c

    a, b, c = coefficients

    # Target value to find hit time
    z_hit = 0  # Adjust this value as needed

    # Solve the quadratic equation: at^2 + bt + (c - z_hit) = 0
    c_prime = c - z_hit
    discriminant = b ** 2 - 4 * a * c_prime

    if discriminant > 0:
        # Two real roots
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        print(f"Hit times: t1 = {t1:.2f}, t2 = {t2:.2f}")
    else:
        return None
    return (polynomial_fit_and_predict(t, x, t2), polynomial_fit_and_predict(t, y, t2))


def polynomial_fit_and_predict(t, x, t1, deg=1):
    coefficients = np.polyfit(t, x, deg)  # Degree 1 for linear fit
    return np.polyval(coefficients, t1)  # Evaluate the polynomial

if __name__ == "__main__":
    PKL = r"pkl/1420.pkl"

    # ____________ If you want to save ______________
    # result_ = merge_id_to_data_maps(FOLDER_PATH)
    # new = create_id_to_cartesian_map(result_)
    # # Save new with pickle into pkl/[time].pkl
    # with open(f"pkl/{1420}.pkl", "wb") as f:
    #     pickle.dump(new, f)

    # ____________ If you want to load ______________
    cartesian_map = load_cartesian_map()

    get_launch_points_clustered(cartesian_map)
    # create_graph.run(map)

    # for a, b in convert_dict_to_list_of_tuples(new):
    #     print(f"ID = {a}")
    #     print(b)
