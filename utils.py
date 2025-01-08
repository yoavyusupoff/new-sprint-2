from typing import Tuple
import math


RADAR_DICT = {
    "Ashdod": (31.77757586390034, 34.65751251836753),
    "Kriyat_Gat": (31.602089287486198, 34.74535762921831),
    "Ofakim": (31.302709659709315, 34.59685294800365),
    "Tseelim": (31.20184656499955, 34.52669152933695),
    "Meron": (33.00023023451869, 35.404698698883585),
    "YABA": (30.653610411909529, 34.783379139342955),
    "Modiin": (31.891980958022323, 34.99481765229601),
    "Gosh_Dan": (32.105913486777084, 34.78624983651992),
    "Carmel": (32.65365306190331, 35.03028065430696)
}


def sphere_to_xyz(radar_name: str, rocket_by_radar: Tuple[float, float, float]) -> Tuple[float, float, float]:
    radar_place = radar_name_to_xy(radar_name)
    x_radar, y_radar = radar_place

    r, phi, theta = rocket_by_radar
    phi = math.radians(phi)
    theta = math.radians(theta)

    x = x_radar + r * math.sin(phi) * math.cos(theta)
    y = y_radar + r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)

    return x, y, z


# set O as Ashdod
def radar_name_to_xy(radar_name: str):
    x, y = RADAR_DICT[radar_name]
    Ox, Oy = RADAR_DICT['Ashdod']
    return x - Ox, y - Oy
