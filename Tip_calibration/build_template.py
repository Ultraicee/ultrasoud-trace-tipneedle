import numpy as np
import math


class build_template(object):
    def __int__(self):
        pass


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
        3Dimension Matrix Group p , P=[p_0,p_1,p_2,p_3] each Group
    :return:
        temp
    """

    a = distance_ab(p[0], p[1])
    c = calp2line(p[0], p[1], p[2])
    b = math.sqrt(math.pow(distance_ab(p[0], p[2]), 2) - math.pow(c, 2))

    for i in range(1, p.size()):
        p[i] -= p[0]

    p[0] = [0, 0, 0]
    r00 = p[1].x / a
    r01 = p[1].y / a
    r02 = p[1].z / a
    r10 = (p[2].x - b * r00) / c
    r11 = (p[2].y - b * r01) / c
    r12 = (p[2].z - b * r02) / c

    r20 = r01 * r12 - r11 * r02
    r21 = r02 * r10 - r00 * r12
    r22 = r00 * r11 - r01 * r10
    R = [r00, r10, r20, r01, r11, r21, r02, r12, r22]

    n = p.size()



