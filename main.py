import numpy as np
import pandas as pd
from create_graph import run  # Assuming the run function is in existing_file.py

from create_graph import *
from csv_to_data import *
from identify_rocket import get_eylon_data
from rocket import *
from utils import *
from utm_to_latlon import utm_to_latlon
from data_to_csv import write_target_bank_result, write_impact_points_result
from test import get_impact_points


def main():
    aylon_result = load_cartesian_map()

    gus_result: List[Tuple[List[int], np.ndarray[float]]] = get_launch_points_clustered(aylon_result)

    lst = []
    for tup in gus_result:
        lat, long = utm_to_latlon(tup[1][0], tup[1][1])
        lst.append((lat, long))

    write_target_bank_result(lst)

    # ramot_result = get_eylon_data(gus_result)

    lst = get_impact_points()
    write_impact_points_result(lst)






if __name__ == "__main__":
    main()
