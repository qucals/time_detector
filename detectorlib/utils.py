import math


def distance_between_points(a_point_1, a_point_2):
    y1, x1 = a_point_1
    y2, x2 = a_point_2

    res = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return res
