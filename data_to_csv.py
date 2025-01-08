from typing import Dict, Tuple, List

import pandas as pd


LATITUDE = "latitude"
LONGITUDE = "longitude"
AMMUNITION = "ammunition"
CERTAINTY = "certainty (True/False)"


def write_impact_points_result(rockets_map: List[Tuple[float, float]]):
    df = pd.DataFrame()

    latitude_lst = []
    longitude_lst = []
    ammunition_lst = []
    certainty_lst = []

    for rocket_data in rockets_map:
        latitude_lst.append(rocket_data[0])
        longitude_lst.append(rocket_data[1])
        ammunition_lst.append("Qassam4")
        certainty_lst.append("True")

    df[LATITUDE] = latitude_lst
    df[LONGITUDE] = longitude_lst
    df[AMMUNITION] = ammunition_lst
    df[CERTAINTY] = certainty_lst

    df.to_csv('impact points.csv', index=False)


def write_target_bank_result(launchers_map: List[Tuple[float, float]]):
    df = pd.DataFrame()

    latitude_lst = []
    longitude_lst = []
    ammunition_lst = []
    certainty_lst = []

    for rocket_data in launchers_map:
        latitude_lst.append(rocket_data[0])
        longitude_lst.append(rocket_data[1])
        ammunition_lst.append("Grad")
        certainty_lst.append("True")

    df[LATITUDE] = latitude_lst
    df[LONGITUDE] = longitude_lst
    df[AMMUNITION] = ammunition_lst
    df[CERTAINTY] = certainty_lst

    df.to_csv('target bank.csv', index=False)


if __name__ == "__main__":
    d = {1: (1, 1, "Kassam"),
         2: (2, 0, "Grad"),
         3: (3, 1, "M75"),
         4: (4, 0, "Grad"),
         5: (5, 1, "Kassam"),
         6: (6, 0, "R160")}
    write_impact_points_result(d)

    l = [(1, 1, "Kassam"),
         (2, 0, "Grad"),
         (3, 1, "M75"),
         (4, 0, "Grad"),
         (5, 1, "Kassam"),
         (6, 0, "R160")]
    write_target_bank_result(l)
