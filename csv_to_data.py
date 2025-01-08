import os
import pandas as pd
from typing import List, Tuple, Dict

from utils import sphere_to_xyz

# ___________________________________________________________________________


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


def get_id_to_data_map(radar_filename: str, folder_path: str) -> Dict[int, pd.DataFrame]:
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


def merge_id_to_data_dicts(folder_path: str) -> List[Tuple[int, pd.DataFrame]]:
    filenames = get_all_filenames(folder_path)
    result_map = dict()

    for radar_filename in filenames:
        # get the mapping of the current radar
        id_to_data_map = get_id_to_data_map(radar_filename, folder_path)

        for rocket_id, data_table in id_to_data_map.items():
            # if the data is of a previously-visited radar, append to the existing data
            if rocket_id in result_map:
                combined = pd.concat([result_map.get(rocket_id), data_table], ignore_index=True)
                result_map[rocket_id] = combined
            else:
                result_map[rocket_id] = data_table

            # sort by the time
            result_map.get(rocket_id).sort_values(by=TIME, ascending=True, ignore_index=True,
                                                  inplace=True)

    return list(sorted(result_map.items()))

# ___________________________________________________________________________


def create_id_to_xyz_table(id_to_data_map: List[Tuple[int, pd.DataFrame]]):
    result = []

    for rocket_id, data_table in id_to_data_map:
        new_data = pd.DataFrame()
        new_data[RADAR_NAME] = data_table[RADAR_NAME]
        new_data[TIME] = data_table[TIME]

        x_list, y_list, z_list = [], [], []

        for index, row in data_table.iterrows():
            r, phi, theta = row[RANGE], row[PHI], row[AZIMUTH]
            x, y, z = sphere_to_xyz(row[RADAR_NAME], (r, phi, theta))
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

        new_data[X] = pd.DataFrame(x_list)
        new_data[Y] = pd.DataFrame(y_list)
        new_data[Z] = pd.DataFrame(z_list)

        result.append((rocket_id, new_data))

    return result


# ___________________________________________________________________________


# run example
if __name__ == "__main__":
    folder_path_ = "./data/With ID/Impact points data"
    result_ = merge_id_to_data_dicts(folder_path_)

    new = create_id_to_xyz_table(result_)

    for tup in new:
        print(f"ID = {tup[0]}")
        print(tup[1])
