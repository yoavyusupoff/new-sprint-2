import geopy.distance

RADAR_DICT = {
    "Ashdod": (31.77757586390034, 34.65751251836753),
    "Kiryat_Gat": (31.602089287486198, 34.74535762921831),
    "Ofakim": (31.302709659709315, 34.59685294800365),
    "Tseelim": (31.20184656499955, 34.52669152933695),
    "Meron": (33.00023023451869, 35.404698698883585),
    "YABA": (30.653610411909529, 34.783379139342955),
    "Modiin": (31.891980958022323, 34.99481765229601),
    "Gosh_Dan": (32.105913486777084, 34.78624983651992),
    "Carmel": (32.65365306190331, 35.03028065430696)
}



O_lat, O_long = RADAR_DICT['Ashdod']

EPSILON = 1e-4
def reverse_pos_long(x, lon_a, lon_b):
    cur = geopy.distance.geodesic((O_lat, lon_a / 2 + lon_b / 2), (O_lat, O_long)).km
    if abs(cur - x) < EPSILON:
        return lon_a / 2 + lon_b / 2
    if cur > x:
        return reverse_pos_long(x, lon_a, (lon_a + lon_b) / 2)
    return reverse_pos_long(x, (lon_a + lon_b) / 2, lon_b)


def reverse_neg_long(x, lon_a, lon_b):
    cur = geopy.distance.geodesic((O_lat, lon_a / 2 + lon_b / 2), (O_lat, O_long)).km
    if abs(cur - x) < EPSILON:
        return lon_a / 2 + lon_b / 2
    if cur > x:
        return reverse_neg_long(x, (lon_a + lon_b) / 2, lon_b)
    return reverse_neg_long(x, lon_a, (lon_a + lon_b) / 2)


def reverse_long(x):
    if x > 0:
        return reverse_pos_long(x, O_long, O_long + 5)
    return reverse_neg_long(-x, O_long - 5, O_long)

def reverse_pos_lat(x, lat_a, lat_b):
    cur = geopy.distance.geodesic((lat_a / 2 + lat_b / 2, O_long), (O_lat, O_long)).km
    if abs(cur - x) < EPSILON:
        return lat_a / 2 + lat_b / 2
    if cur > x:
        return reverse_pos_lat(x, lat_a, (lat_a + lat_b) / 2)
    return reverse_pos_lat(x, (lat_a + lat_b) / 2, lat_b)


def reverse_neg_lat(x, lat_a, lat_b):
    cur = geopy.distance.geodesic((lat_a / 2 + lat_b / 2, O_long), (O_lat, O_long)).km
    if abs(cur - x) < EPSILON:
        return lat_a / 2 + lat_b / 2
    if cur > x:
        return reverse_neg_lat(x, (lat_a + lat_b) / 2, lat_b)
    return reverse_neg_lat(x, lat_a, (lat_a + lat_b) / 2)


def reverse_lat(x):
    if x > 0:
        return reverse_pos_lat(x, O_lat, O_lat + 5)
    return reverse_neg_lat(-x, O_lat - 5, O_lat)


def utm_to_latlon(utm_x, utm_y):
    lon = reverse_long(utm_x)
    lat = reverse_lat(utm_y)
    return lat, lon

