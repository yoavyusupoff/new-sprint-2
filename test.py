from typing import Tuple, List

import utm_to_latlon
from csv_to_data import merge_id_to_data_maps, create_id_to_cartesian_map, get_landing_point

X = "x"
Y = "y"
Z = "z"

TIME = "time"
def get_impact_points() -> List[Tuple[float, float]]:
    FOLDER_PATH = "./data/With ID/Impact points data"

    # ____________ If you want to save ______________
    result_ = merge_id_to_data_maps(FOLDER_PATH)
    dic = create_id_to_cartesian_map(result_)
    all_launches = []
    launches_id = []
    for rocket_id, mat in dic.items():
        mat = dic[rocket_id]
        t = mat[TIME]
        x = mat[X]
        y = mat[Y]
        z = mat[Z]
        if get_landing_point(x, y, z, t) is None:
            continue
        x_l, y_l = get_landing_point(x, y, z, t)
        launches_id.append(rocket_id)
        all_launches.append(utm_to_latlon.utm_to_latlon(x_l, y_l))
    return all_launches
