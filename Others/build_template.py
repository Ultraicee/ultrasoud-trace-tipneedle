import numpy as np
import math

import scipy


class template:
    """
    description:
        建造针尖模板坐标系。详见技术手册
    """

    def __init__(self, measure3d):
        """
        description: 初始化
        :param measure3d: 相机坐标系下的测量信息 
        """
        # 没有排序，采集到的不同位置的小球数据
        self.P0 = measure3d[:, :, 0]
        self.P1 = measure3d[:, :, 1]
        self.P2 = measure3d[:, :, 2]
        self.P3 = measure3d[:, :, 3]

        # template P
        self.Pt_0 = 0
        self.Pt_1 = 0
        self.Pt_2 = 0
        self.Pt_3 = 0

        # flag
        self.p_flag = [0, 0, 0, 0]

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
            计算三维点AB的二范数，（弃用）
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

    def calTriangle(self, a, b, c):
        """
        description:
            计算三个三维点组成的三角形面积。面积越大的三角形所对应的向量的叉积的长度也越大。
        :param a: 顶点 a
        :param b: 顶点 b
        :param c: 顶点 c
        :return:
            area 三角形面积
        """
        v1 = b - a
        v2 = c - a
        cross_product = np.cross(v1, v2)  # 计算向量的叉积
        area = 0.5 * scipy.linalg.norm(cross_product)
        return area

    def reorder(self):
        """
        description:
            对收集到的小球数按照模板要求重新排序。

        :return:
        """

        p = [0, 0, 0, 0]
        p[0] = self.P0.copy()
        p[1] = self.P1.copy()
        p[2] = self.P2.copy()
        p[3] = self.P3.copy()
        coutFlag_pt1 = 0
        coutFlag_pt2 = 0

        # print(p[0], p[1], p[2], p[3])

        # 计算四个小球的中心点
        center = np.mean(np.array([p[0], p[1], p[2], p[3]]), axis=0)
        dis1 = scipy.linalg.norm(abs(p[0] - center))
        dis2 = scipy.linalg.norm(abs(p[1] - center))
        dis3 = scipy.linalg.norm(abs(p[2] - center))
        dis4 = scipy.linalg.norm(abs(p[3] - center))
        # print(dis1, dis2, dis3, dis4)

        # 确定Pt_0
        min_index = np.argmin([dis1, dis2, dis3, dis4])  # 返回最小值的下标索引
        print("give me the min: ", min_index)
        self.p_flag[min_index] = 1
        self.Pt_0 = p[min_index]
        print("P0 has reorder")

        # 确认 Pt_1
        max_line = 0

        for i in range(4):
            if self.p_flag[i]:
                continue
            line = scipy.linalg.norm(abs(self.Pt_0 - p[i]), 2)
            if line > max_line:
                max_line = line
                coutFlag_pt1 = i
                self.Pt_1 = p[i]
        self.p_flag[coutFlag_pt1] = 1

        # 确认 Pt_2
        max_line2 = 0
        for i in range(4):
            if self.p_flag[i]:
                # 已经确认的点将会跳过
                continue
            line2 = self.calp2line(self.Pt_0[0], self.Pt_1[0], p[i][0])
            if line2 > max_line2:
                max_line2 = line2
                coutFlag_pt2 = i
                self.Pt_2 = p[i]
        self.p_flag[coutFlag_pt2] = 1

        # 确认前面三个点之后，直接进行赋值
        for i in range(4):
            if self.p_flag[i]:
                continue
            self.Pt_3 = p[i]

    def Template_build(self):
        """
        description:
            输入排好序的4个小球
            按照规则制作初始化矩阵模板。
            每次标定的时候取同一组的数据中第一帧作为初始模板即可
        :param:
            reorder 3Dimension Matrix Group p , P=[p_0,p_1,p_2,p_3] each Group
        :return:
            temp
        """

        # a = self.distance_ab(self.P0[0], self.P1[0])
        a = scipy.linalg.norm(abs(self.Pt_0[0] - self.Pt_1[0]), 2)
        c = self.calp2line(self.Pt_2[0], self.Pt_0[0], self.Pt_1[0])
        b = math.sqrt(math.pow((scipy.linalg.norm(abs(self.Pt_0[0] - self.Pt_2[0]), 2)), 2) - math.pow(c, 2))
        print(a, b, c)
        # 对P1,P2,P3 相对于P0进行偏移
        self.Pt_1[0] -= self.Pt_0[0]
        self.Pt_2[0] -= self.Pt_0[0]
        self.Pt_3[0] -= self.Pt_0[0]

        P1_x, P1_y, P1_z = self.xyz(self.Pt_1[0])
        P2_x, P2_y, P2_z = self.xyz(self.Pt_2[0])
        P3_x, P3_y, P3_z = self.xyz(self.Pt_3[0])

        self.Pt_0[0] = [0, 0, 0]
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

        R_inv = np.linalg.inv(R)
        line_inv = np.linalg.inv(line)
        # print(R_inv)
        temp = np.matmul(R_inv, line_inv)

        return temp
