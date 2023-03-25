import numpy as np
import math

import scipy


class template:
    """
    description:
        建造针尖模板坐标系。详见技术手册

    """

    def __init__(self, Matrix3d):
        self.P0 = Matrix3d[:, :, 0]
        self.P1 = Matrix3d[:, :, 1]
        self.P2 = Matrix3d[:, :, 2]
        self.P3 = Matrix3d[:, :, 3]
        # print(self.P0, self.P1, self.P2, self.P3)

    def xyz(self, P):
        """
        description:
            拆开每个点的三维值

        """
        P_x = P[0]
        P_y = P[1]
        P_z = P[2]

        return P_x, P_y, P_z

    def distance_ab(self, a, b):
        """
        description:
            计算三维点AB的二范数
        :param:
            3Dimension A，B
        :return:
            2-Norm fo AB
        """
        A_x = a[0]
        A_y = a[1]
        A_z = a[2]

        B_x = b[0]
        B_y = b[1]
        B_z = b[2]
        return math.sqrt((A_x - B_x) * (A_x - B_x) + (A_y - B_y) * (A_y - B_y) + (A_z - B_z) * (A_z - B_z))

    def calp2line(self, a, b, p):
        """
        description:
            计算三维点P到AB组成的直线的距离
        :param:
            3Dimension A，B，P
        :return:
            Distance
        """

        A_x = a[0]
        A_y = a[1]
        A_z = a[2]
        B_x = b[0]
        B_y = b[1]
        B_z = b[2]
        P_x = p[0]
        P_y = p[1]
        P_z = p[2]
        ##
        # abc = abs(b - c)
        # di s2 = scipy.linalg.norm(abc, 2)
        ##
        P = math.sqrt((math.pow((A_x - B_x), 2) + math.pow((A_y - B_y), 2) + math.pow((A_z - B_z), 2)))
        B = math.sqrt((math.pow((A_x - P_x), 2) + math.pow((A_y - P_y), 2) + math.pow((A_z - P_z), 2)))
        A = math.sqrt((math.pow((P_x - B_x), 2) + math.pow((P_y - B_y), 2) + math.pow((P_z - B_z), 2)))
        cosp = (math.pow(A, 2) + math.pow(B, 2) - math.pow(P, 2)) / (2 * A * B)
        sinp = math.sqrt(1 - math.pow(cosp, 2))
        return A * B * sinp / P

    def Template_build(self):
        """
        description:
            按照规则制作4个小球的初始化矩阵模板。
            每次标定的时候取同一组的数据中第一帧作为初始模板即可
        :param:
            3Dimension Matrix Group p , P=[p_0,p_1,p_2,p_3] each Group
        :return:
            temp
        """

        # a = self.distance_ab(self.P0[0], self.P1[0])
        a = scipy.linalg.norm(abs(self.P0[0] - self.P1[0]), 2)
        c = self.calp2line(self.P2[0], self.P0[0], self.P1[0])
        b = math.sqrt(math.pow((scipy.linalg.norm(abs(self.P0[0] - self.P2[0]), 2)), 2) - math.pow(c, 2))
        print(a, b, c)
        # 对P1,P2,P3 相对于P0进行偏移
        self.P1[0] -= self.P0[0]
        self.P2[0] -= self.P0[0]
        self.P3[0] -= self.P0[0]

        P1_x, P1_y, P1_z = self.xyz(self.P1[0])
        P2_x, P2_y, P2_z = self.xyz(self.P2[0])
        P3_x, P3_y, P3_z = self.xyz(self.P3[0])

        self.P0[0] = [0, 0, 0]
        r00 = P1_x / a
        r01 = P1_y / a
        r02 = P1_z / a
        r10 = (P2_x - b * r00) / c
        r11 = (P2_y - b * r01) / c
        r12 = (P2_z - b * r02) / c
        # outer r20,r21,r22
        r20 = r01 * r12 - r11 * r02
        r21 = r02 * r10 - r00 * r12
        r22 = r00 * r11 - r01 * r10

        R = np.array([[r00, r10, r20], [r01, r11, r21], [r02, r12, r22]])
        # print(R)
        line = np.array([[P1_x, P1_y, P1_z], [P2_x, P2_y, P2_z], [P3_x, P3_y, P3_z]])

        R_inv = R.T
        line_inv = line.T
        # print(R_inv)
        temp = R_inv * line_inv

        return temp
