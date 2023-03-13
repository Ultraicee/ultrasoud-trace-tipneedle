import numpy as np
import math


def distance_ab(a, b):
    """
    description:
        计算三维点AB的二范数
    :param:
        3Dimension A，B
    :return:
        2-Norm fo AB
    """
    A_x = a[:, 0, 0]
    A_y = a[0, :, 0]
    A_z = a[0, 0, :]
    B_x = b[:, 0, 0]
    B_y = b[0, :, 0]
    B_z = b[0, 0, :]
    return math.sqrt((A_x - B_x) * (A_x - B_x) + (A_y - B_y) * (A_y - B_y) + (A_z - B_z) * (A_z - B_z))


def calp2line(a, b, p):
    """
    description:
        计算三维点P到AB组成的直线的距离
    :param:
        3Dimension A，B，P
    :return:
        Distance
    """
    A_x = a[:, 0, 0]
    A_y = a[0, :, 0]
    A_z = a[0, 0, :]
    B_x = b[:, 0, 0]
    B_y = b[0, :, 0]
    B_z = b[0, 0, :]
    P_x = p[:, 0, 0]
    P_y = p[0, :, 0]
    P_z = p[0, 0, :]
    P = math.sqrt((math.pow((A_x - B_x), 2) + math.pow((A_y - B_y), 2) + math.pow((A_z - B_z), 2)))
    B = math.sqrt((math.pow((A_x - P_x), 2) + math.pow((A_y - P_y), 2) + math.pow((A_z - P_z), 2)))
    A = math.sqrt((math.pow((P_x - B_x), 2) + math.pow((P_y - B_y), 2) + math.pow((P_z - B_z), 2)))
    cosp = (math.pow(A, 2) + math.pow(B, 2) - math.pow(P, 2)) / (2 * A * B)
    sinp = math.sqrt(1 - math.pow(cosp, 2))
    return A * B * sinp / P


def Template_build(p):
    """
    description:
        按照一定规则制作4个小球的初始化矩阵模板。
        每次标定的时候取同一组的数据中第一帧作为初始模板即可
    :param:
        3Dimension Matrix Group p
    :return:
        temp
    """
