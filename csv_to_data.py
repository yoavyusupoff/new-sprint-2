import os

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import tqdm

import create_graph
from utils import sphere_to_xyz
import pickle
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

    # copy the radar name and time columns
    result[RADAR_NAME] = table[RADAR_NAME]
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


def main(folder_path: str) -> Dict[int, np.ndarray]:
    merged = merge_id_to_data_maps(folder_path)
    merged_in_cartesian = create_id_to_cartesian_map(merged)

    result = dict()
    for key, id_map in convert_dict_to_list_of_tuples(merged_in_cartesian):
        result[key] = id_map.to_numpy()

    return result


if __name__ == "__main__":
    folder_path_ = "./data/With ID/Target bank data"

    PKL = r"pkl/1341.pkl"
    # ____________ If you want to save ______________
    # result_ = merge_id_to_data_maps(folder_path_)
    # new = create_id_to_cartesian_map(result_)
    # Save new with pickle into pkl/[time].pkl
    # with open(f"pkl/{1333}.pkl", "wb") as f:
    #     pickle.dump(new, f)

    # ____________ If you want to load ______________
    with open(PKL, "rb") as f:
        new = pickle.load(f)


    create_graph.run(map)
