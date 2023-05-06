import os
import numpy as np
import math
import scipy
from Others.kabsch import kabsch  # 使用kabsch函数计算旋转平移矩阵
# 尝试利用pytorch方法解决雅可比矩阵问题(弃用)
from torch.autograd.functional import jacobian
# 引入优化器相关库
import torch
import torch.nn as nn
import torch.optim as optim

from Others.yaml_create import yaml_handle


def isRotationMatrix(R):
    """
    description:
        检查输入的矩阵是否符合欧拉角的条件。
    :param R: 旋转矩阵
    :return: n
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """
    description:
        将旋转矩阵转换成欧拉角（弧度），除了排列顺序之外（x和z的顺序），结果和matlab的一致
    :param R: 旋转矩阵
    :return:角度值 - x,y,z
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(alpha, beta, gamma):
    """
    description:
        将欧拉角转换成旋转矩阵
    :param: theta: [x,y,z]
    :return: R: shape(3,3) 旋转矩阵 type:numpy.ndarray
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(alpha), -math.sin(alpha)],
                    [0, math.sin(alpha), math.cos(alpha)]
                    ])

    R_y = np.array([[math.cos(beta), 0, math.sin(beta)],
                    [0, 1, 0],
                    [-math.sin(beta), 0, math.cos(beta)]
                    ])

    R_z = np.array([[math.cos(gamma), -math.sin(gamma), 0],
                    [math.sin(gamma), math.cos(gamma), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


class template:
    """
    description:
        建造针尖模板坐标系。
    """

    def __init__(self, measure3d):
        """
        description:
        初始化
        :param:
        measure3d: 相机坐标系下的测量信息
        """
        # 没有排序，采集到的不同位置的小球数据
        self.P0 = measure3d[:, :, 0]
        self.P1 = measure3d[:, :, 1]
        self.P2 = measure3d[:, :, 2]
        self.P3 = measure3d[:, :, 3]

        # 采集数据的张数
        self.Fig_N = len(measure3d)

        # 排序之后的小球数据
        self.reorder_P0 = None
        self.reorder_P1 = None
        self.reorder_P2 = None
        self.reorder_P3 = None

        # 模板坐标系Pt各点坐标
        self.Pt_0 = None
        self.Pt_1 = None
        self.Pt_2 = None
        self.Pt_3 = None

        # 制作模板时候使用的flag
        self.p_flag = [0, 0, 0, 0]

        # RT旋转平移矩阵temp
        self.R = []
        self.t = []

        # 欧拉角(由旋转矩阵转换而来)
        self.alpha = None
        self.beta = None
        self.gamma = None
        # 平移向量
        self.T1 = None
        self.T2 = None
        self.T3 = None

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

    def Template_PointReorder(self):
        """
        description:
            对收集到的小球数按照模板要求重新排序。

        """

        p = [0, 0, 0, 0]
        p[0] = self.P0.copy()
        p[1] = self.P1.copy()
        p[2] = self.P2.copy()
        p[3] = self.P3.copy()
        coutFlag_pt1 = 0
        coutFlag_pt2 = 0

        # 计算四个小球的中心点
        center = np.mean(np.array([p[0], p[1], p[2], p[3]]), axis=0)
        dis1 = np.linalg.norm(abs(p[0] - center))
        dis2 = np.linalg.norm(abs(p[1] - center))
        dis3 = np.linalg.norm(abs(p[2] - center))
        dis4 = np.linalg.norm(abs(p[3] - center))
        # print(dis1, dis2, dis3, dis4)

        # 确定Pt_0
        min_index = np.argmin([dis1, dis2, dis3, dis4])  # 返回最小值的下标索引
        # print("give me the min: ", min_index)
        self.p_flag[min_index] = 1
        self.reorder_P0 = p[min_index]
        # print("P0 has reorder")

        # 确认 Pt_1
        max_line = 0
        for i in range(4):
            if self.p_flag[i]:
                continue
            line = np.linalg.norm(abs(self.reorder_P0 - p[i]), 2)
            if line > max_line:
                max_line = line
                coutFlag_pt1 = i
                self.reorder_P1 = p[i]
        self.p_flag[coutFlag_pt1] = 1

        # 确认 reorder_P2
        max_line2 = 0
        for i in range(4):
            if self.p_flag[i]:
                # 已经确认的点将会跳过
                continue
            line2 = self.calp2line(self.reorder_P0[0], self.reorder_P1[0], p[i][0])
            if line2 > max_line2:
                max_line2 = line2
                coutFlag_pt2 = i
                self.reorder_P2 = p[i]
        self.p_flag[coutFlag_pt2] = 1

        # 确认前面三个点之后，直接进行赋值
        for i in range(4):
            if self.p_flag[i]:
                continue
            self.reorder_P3 = p[i]

        print("Balls has ordered.")

    def Template_initBuild(self, N):
        """
        description:
            输入排好序的4个小球
            按照规则制作初始化矩阵模板。
            每次标定的时候取同一组的数据中第一帧作为初始模板即可,剩下的交给模板优化函数

        :param:
            reorder 3Dimension Matrix Group p , P=[p_0,p_1,p_2,p_3] each Group
            N：the N frames.
        :return:
            shape为(3,4)的初始化模板坐标系矩阵
        """

        # a = self.distance_ab(self.P0[0], self.P1[0])
        a = np.linalg.norm(abs(self.reorder_P0[N] - self.reorder_P1[N]), 2)
        c = self.calp2line(self.reorder_P0[N], self.reorder_P1[N], self.reorder_P2[N])
        b = math.sqrt(
            math.pow((np.linalg.norm(abs(self.reorder_P0[N] - self.reorder_P2[N]), 2)), 2) - math.pow(c, 2))
        # print(a, b, c)
        # 对P1,P2,P3 相对于P0进行偏移
        Pt_temp1 = self.reorder_P1[N].copy() - self.reorder_P0[N].copy()
        Pt_temp2 = self.reorder_P2[N].copy() - self.reorder_P0[N].copy()
        Pt_temp3 = self.reorder_P3[N].copy() - self.reorder_P0[N].copy()
        Pt_temp0 = [0, 0, 0]
        P0_x, P0_y, P0_z = self.xyz(Pt_temp0)
        P1_x, P1_y, P1_z = self.xyz(Pt_temp1)
        P2_x, P2_y, P2_z = self.xyz(Pt_temp2)
        P3_x, P3_y, P3_z = self.xyz(Pt_temp3)

        r00 = P1_x / a
        r01 = P1_y / a
        r02 = P1_z / a
        r10 = (P2_x - b * r00) / c
        r11 = (P2_y - b * r01) / c
        r12 = (P2_z - b * r02) / c

        # R旋转矩阵第三列 r20,r21,r22（？）
        r20 = r01 * r12 - r11 * r02
        r21 = r02 * r10 - r00 * r12
        r22 = r00 * r11 - r01 * r10

        R = np.array([[r00, r10, r20], [r01, r11, r21], [r02, r12, r22]])

        line = np.array([[P0_x, P0_y, P0_z], [P1_x, P1_y, P1_z], [P2_x, P2_y, P2_z], [P3_x, P3_y, P3_z]])

        R_inv = np.linalg.inv(R)  # R的逆
        line_inv = line.T  # T的转置

        temp = np.matmul(R_inv, line_inv)  # 矩阵相乘
        temp_t = temp.T
        self.Pt_0 = temp_t[0]
        self.Pt_1 = temp_t[1]
        self.Pt_2 = temp_t[2]
        self.Pt_3 = temp_t[3]

        return temp

    def Matrix_RT_Conversion(self, N):
        """
        description:
            利用kabsch函数计算初始模板坐标系和测量数据的平移旋转矩阵,将旋转矩阵转换成欧拉角（弧度制），
            再用xyz函数分别拆成单独的欧拉角和平移值
        :param N: 第N组测量数据
        :return: 3个欧拉角，3个平移值
        """
        Measure = np.array([self.reorder_P0[N], self.reorder_P1[N], self.reorder_P2[N], self.reorder_P3[N]])
        template_N = np.array([self.Pt_0, self.Pt_1, self.Pt_2, self.Pt_3])
        R, t = kabsch(Measure, template_N)
        alpha, beta, gamma = self.xyz(rotationMatrixToEulerAngles(R))
        T1, T2, T3 = self.xyz(t)
        return alpha, beta, gamma, T1, T2, T3  # 正常使用
        # return R, t, alpha, beta, gamma, T1, T2, T3  # test用

    def theta_Dataproc(self):
        """

        :return:待优化的12个参数（6个模板总使用的参数，3个欧拉角，3个平移向量）
        """
        N = self.Fig_N
        alpha = np.zeros((N, 1))
        beta = np.zeros((N, 1))
        gamma = np.zeros((N, 1))
        T1 = np.zeros((N, 1))
        T2 = np.zeros((N, 1))
        T3 = np.zeros((N, 1))
        for i in range(N):
            alpha[i], beta[i], gamma[i], T1[i], T2[i], T3[i] = self.Matrix_RT_Conversion(i)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

        a = self.Pt_1[0].copy()
        b = self.Pt_2[0].copy()
        c = self.Pt_2[1].copy()
        d = self.Pt_3[0].copy()
        e = self.Pt_3[1].copy()
        f = self.Pt_3[2].copy()

        # 将测量数据整合成（12*N，1）的格式，排列为[p1x,p1y,p1z,p2x,p2y,p2z,p3x,p3y,p3z,p4x,p4y,p4z..(循环N组)]
        p0 = self.reorder_P0.copy().reshape(3 * N, 1)
        p1 = self.reorder_P1.copy().reshape(3 * N, 1)
        p2 = self.reorder_P2.copy().reshape(3 * N, 1)
        p3 = self.reorder_P3.copy().reshape(3 * N, 1)
        P_M = []
        for i in range(N):
            P_M.append(p0[i * 3])
            P_M.append(p0[i * 3 + 1])
            P_M.append(p0[i * 3 + 2])
            P_M.append(p1[i * 3])
            P_M.append(p1[i * 3 + 1])
            P_M.append(p1[i * 3 + 2])
            P_M.append(p2[i * 3])
            P_M.append(p2[i * 3 + 1])
            P_M.append(p2[i * 3 + 2])
            P_M.append(p3[i * 3])
            P_M.append(p3[i * 3 + 1])
            P_M.append(p3[i * 3 + 2])
        P_M = np.array(P_M).reshape(12 * N, 1)

        return a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3, P_M

    def cost_function(self, a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3):
        """
        description:
            梯度下降优化里的损失函数
        :param f:
        :param e:
        :param d:
        :param c:
        :param b:
        :param a:
        :param alpha: 欧拉角alpha
        :param beta: 欧拉角beta
        :param gamma: 欧拉角gamma
        :param T1:平移向量1
        :param T2:平移向量2
        :param T3:平移向量3
        :return: type:tuple
        """

        N = self.Fig_N

        p0 = self.reorder_P0.copy().reshape(3 * N, 1)
        p1 = self.reorder_P1.copy().reshape(3 * N, 1)
        p2 = self.reorder_P2.copy().reshape(3 * N, 1)
        p3 = self.reorder_P3.copy().reshape(3 * N, 1)

        # p0 = torch.tensor(p0)
        # p1 = torch.tensor(p1)
        # p2 = torch.tensor(p2)
        # p3 = torch.tensor(p3)

        # E = torch.zeros(N * 12, 1)
        E = np.zeros((N * 12, 1))  # 预测值向量
        J = np.zeros((N * 12, 6 + 6 * N))  # 预测值向量的雅可比矩阵，rows:12N,cols:6+6N(6为模板矩阵6个参数，6N为N组3个欧拉角和3个平移向量组合起来）
        """
         由于通过预测值向量求取的雅可比矩阵是个很大的稀疏矩阵，只需要索引出非零值即可。
        """
        for i in range(N):
            """
            构建损失函数
            """
            E[i * 12] = T1[i] - p0[i * 3]
            E[i * 12 + 1] = T2[i] - p0[i * 3 + 1]
            E[i * 12 + 2] = T3[i] - p0[i * 3 + 2]

            E[i * 12 + 3] = T1[i] - p1[i * 3] + a * math.cos(alpha[i]) * math.cos(beta[i])
            E[i * 12 + 4] = T2[i] - p1[i * 3 + 1] - a * (
                    math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
                gamma[i]))
            E[i * 12 + 5] = T3[i] - p1[i * 3 + 2] + a * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i]))

            E[i * 12 + 6] = T1[i] - p2[i * 3] + b * math.cos(alpha[i]) * math.cos(beta[i]) + c * math.cos(
                beta[i]) * math.sin(alpha[i])
            E[i * 12 + 7] = T2[i] - p2[i * 3 + 1] - b * (
                    math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
                gamma[i])) + c * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i]))
            E[i * 12 + 8] = T3[i] - p2[i * 3 + 2] + b * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i])) - c * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
                alpha[i]) * math.sin(beta[i]))

            E[i * 12 + 9] = T1[i] - p3[i * 3] - f * math.sin(beta[i]) + d * math.cos(alpha[i]) * math.cos(
                beta[i]) + e * math.cos(beta[i]) * math.sin(alpha[i])
            E[i * 12 + 10] = T2[i] - p3[i * 3 + 1] - d * (
                    math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
                gamma[i])) + e * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i])) + f * math.cos(beta[i]) * math.sin(gamma[i])
            E[i * 12 + 11] = T3[i] - p3[i * 3 + 2] + d * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i])) - e * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
                alpha[i]) * math.sin(beta[i])) + f * math.cos(beta[i]) * math.cos(gamma[i])

            """ 
            构建损失函数的雅可比矩阵
            rows->3:8   cols->0:2
            """
            J[12 * i + 3, 0] = math.cos(alpha[i]) * math.cos(beta[i])
            J[12 * i + 4, 0] = math.cos(alpha[i]) * math.cos(beta[i]) * math.sin(gamma[i]) - math.cos(
                gamma[i]) * math.sin(alpha[i])
            J[12 * i + 5, 0] = math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i])
            J[12 * i + 6, 1] = math.cos(alpha[i]) * math.cos(beta[i])
            J[12 * i + 6, 2] = math.cos(beta[i]) * math.sin(alpha[i])
            J[12 * i + 7, 1] = math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(gamma[i]) - math.cos(
                gamma[i]) * math.sin(alpha[i])
            J[12 * i + 7, 2] = math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i])

            J[12 * i + 8, 1] = math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i])
            J[12 * i + 8, 2] = math.cos(gamma[i]) * math.sin(alpha[i]) * math.sin(beta[i]) - math.cos(
                alpha[i]) * math.sin(
                gamma[i])

            """
            rows->0:8 cols->6:6+5
            """

            J[12 * i, 6 + 6 * i + 3] = 1
            J[12 * i + 1, 6 + 6 * i + 4] = 1
            J[12 * i + 2, 6 + 6 * i + 5] = 1
            J[12 * i + 3, 6 + 6 * i] = -math.cos(beta[i]) * math.sin(alpha[i]) * a
            J[12 * i + 3, 6 + 6 * i + 1] = -math.cos(alpha[i]) * math.sin(beta[i]) * a
            J[12 * i + 3, 6 + 6 * i + 3] = 1
            J[12 * i + 4, 6 + 6 * i] = -a * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i]))
            J[12 * i + 4, 6 + 6 * i + 1] = math.cos(alpha[i]) * math.cos(beta[i]) * math.sin(gamma[i]) * a
            J[12 * i + 4, 6 + 6 * i + 2] = a * (math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i]))
            J[12 * i + 4, 6 + 6 * i + 4] = 1
            J[12 * i + 5, 6 + 6 * i] = a * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.sin(alpha[i]) * math.sin(
                beta[i]) * math.cos(gamma[i]))
            J[12 * i + 5, 6 + 6 * i + 1] = math.cos(alpha[i]) * math.cos(beta[i]) * math.cos(gamma[i]) * a
            J[12 * i + 5, 6 + 6 * i + 2] = a * (math.sin(alpha[i]) * math.cos(gamma[i]) - math.cos(alpha[i]) * math.sin(
                gamma[i]) * math.sin(beta[i]))
            J[12 * i + 5, 6 + 6 * i + 5] = 1
            J[12 * i + 6, 6 + 6 * i] = math.cos(alpha[i]) * math.cos(beta[i]) * c - math.cos(beta[i]) * math.sin(
                alpha[i]) * b
            J[12 * i + 6, 6 + 6 * i + 1] = -math.cos(alpha[i]) * math.sin(beta[i]) * b - math.sin(alpha[i]) * math.sin(
                beta[i]) * c
            J[12 * i + 6, 6 + 6 * i + 3] = 1
            J[12 * i + 7, 6 + 6 * i] = -b * (
                    math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(alpha[i]) * math.sin(beta[i]) * math.sin(
                gamma[i])) - c * ((math.cos(gamma[i]) * math.sin(alpha[i])) - math.cos(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i]))
            J[12 * i + 7, 6 + 6 * i + 1] = math.cos(alpha[i]) * math.cos(beta[i]) * math.sin(gamma[i]) * b + math.cos(
                beta[i]) * math.sin(alpha[i]) * math.sin(gamma[i]) * c
            J[12 * i + 7, 6 + 6 * i + 2] = b * (math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i])) - c * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(
                gamma[i]) * math.sin(alpha[i]) * math.sin(beta[i]))
            J[12 * i + 7, 6 + 6 * i + 4] = 1
            J[12 * i + 8, 6 + 6 * i] = b * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.sin(alpha[i]) * math.sin(
                beta[i]) * math.cos(gamma[i])) + c * (math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(
                alpha[i]) * math.cos(gamma[i]) * math.sin(beta[i]))
            J[12 * i + 8, 6 + 6 * i + 1] = b * math.cos(alpha[i]) * math.cos(beta[i]) * math.cos(gamma[i]) + math.cos(
                beta[i]) * math.cos(gamma[i]) * math.sin(alpha[i]) * c
            J[12 * i + 8, 6 + 6 * i + 2] = b * (math.sin(alpha[i]) * math.cos(gamma[i]) - math.cos(alpha[i]) * math.sin(
                gamma[i]) * math.sin(beta[i])) - c * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(
                alpha[i]) * math.sin(beta[i]) * math.sin(gamma[i]))
            J[12 * i + 8, 6 + 6 * i + 5] = 1

            """
            rows->9:11 cols->3:5
            """
            J[12 * i + 3 * 4 - 3, 3 * 4 - 9] = math.cos(alpha[i]) * math.cos(beta[i])
            J[12 * i + 3 * 4 - 3, 3 * 4 - 8] = math.cos(beta[i]) * math.sin(alpha[i])
            J[12 * i + 3 * 4 - 3, 3 * 4 - 7] = -math.sin(beta[i])
            J[12 * i + 3 * 4 - 3, 6 + 6 * i] = math.cos(alpha[i]) * math.cos(beta[i]) * e - math.cos(
                beta[i]) * math.sin(alpha[i]) * d
            J[12 * i + 3 * 4 - 3, 6 + 6 * i + 1] = -math.cos(beta[i]) * f - math.cos(alpha[i]) * math.sin(
                beta[i]) * d - math.sin(alpha[i]) * math.sin(beta[i]) * e
            J[12 * i + 3 * 4 - 3, 6 + 6 * i + 3] = 1
            J[12 * i + 3 * 4 - 2, 3 * 4 - 9] = math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(gamma[i]) - math.cos(
                gamma[i]) * math.sin(alpha[i])
            J[12 * i + 3 * 4 - 2, 3 * 4 - 8] = math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i])
            J[12 * i + 3 * 4 - 2, 3 * 4 - 7] = math.cos(beta[i]) * math.sin(gamma[i])
            J[12 * i + 3 * 4 - 2, 6 + 6 * i] = -d * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(
                alpha[i]) * math.sin(beta[i]) * math.sin(gamma[i])) - e * (
                                                       math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(
                                                   alpha[i]) * math.sin(beta[i]) * math.sin(gamma[i]))
            J[12 * i + 3 * 4 - 2, 6 + 6 * i + 1] = math.cos(alpha[i]) * math.cos(beta[i]) * math.sin(
                gamma[i]) * d - math.sin(beta[i]) * math.sin(gamma[i]) * f + math.cos(beta[i]) * math.sin(
                alpha[i]) * math.sin(gamma[i]) * e
            J[12 * i + 3 * 4 - 2, 6 + 6 * i + 2] = d * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i])) - e * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
                alpha[i]) * math.sin(beta[i])) + f * math.cos(beta[i]) * math.cos(gamma[i])
            J[12 * i + 3 * 4 - 2, 6 + 6 * i + 4] = 1
            J[12 * i + 3 * 4 - 1, 3 * 4 - 9] = math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i])
            J[12 * i + 3 * 4 - 1, 3 * 4 - 8] = math.cos(gamma[i]) * math.sin(alpha[i]) * math.sin(beta[i]) - math.cos(
                alpha[i]) * math.sin(gamma[i])
            J[12 * i + 3 * 4 - 1, 3 * 4 - 7] = math.cos(beta[i]) * math.cos(gamma[i])
            J[12 * i + 3 * 4 - 1, 6 + 6 * i] = d * (
                    math.cos(alpha[i]) * math.sin(gamma[i]) - math.sin(alpha[i]) * math.sin(beta[i]) * math.cos(
                gamma[i])) + e * (math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(
                gamma[i]) * math.sin(beta[i]))
            J[12 * i + 3 * 4 - 1, 6 + 6 * i + 1] = math.cos(alpha[i]) * math.cos(beta[i]) * math.cos(
                gamma[i]) * d - math.cos(gamma[i]) * math.sin(beta[i]) * f + math.cos(beta[i]) * math.cos(
                gamma[i]) * math.sin(alpha[i]) * e
            J[12 * i + 3 * 4 - 1, 6 + 6 * i + 2] = d * (
                    math.sin(alpha[i]) * math.cos(gamma[i]) - math.cos(alpha[i]) * math.sin(gamma[i]) * math.sin(
                beta[i])) - e * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i])) - f * math.cos(beta[i]) * math.sin(gamma[i])
            J[12 * i + 3 * 4 - 1, 6 + 6 * i + 5] = 1

        # 计算梯度：
        grad = -E.T @ J
        return grad

    def Template_OPT(self):
        """
        梯度下降算法主要函数
        :return:
        """
        N = self.Fig_N
        # 将模板坐标系的初值提取出来，并给参数添加一点偏置。
        a = self.Pt_1[0].copy() + 1
        b = self.Pt_2[0].copy() + 1
        c = self.Pt_2[1].copy() - 1
        d = self.Pt_3[0].copy() - 1
        e = self.Pt_3[1].copy() - 1
        f = self.Pt_3[2].copy() - 1

        # 构建测量值矩阵
        P_test = []
        for i in range(N):
            temp = np.array([self.reorder_P0[i], self.reorder_P1[i], self.reorder_P2[i], self.reorder_P3[i]])
            P_test.append(temp.T)
        P_test = np.array(P_test).reshape(N * 3, 4)
        # 给欧拉角和平移向量添加一点偏置
        for i in range(N):
            self.alpha[i] += 0.1
            self.beta[i] -= 0.1
            self.gamma[i] += 0.1
            self.T1[i] += 1
            self.T2[i] -= 1
            self.T3[i] += 1

        epoch = 0
        delta_p = np.zeros((1, 6 + 6 * N))  # 梯度值初始化
        while epoch < 5000:
            delta_p = self.cost_function(a, b, c, d, e, f, self.alpha, self.beta, self.gamma, self.T1, self.T2, self.T3)
            # 更新模板6个参数的值，学习率为0.01

            a += 0.00001 * delta_p[0, 0]
            b += 0.00001 * delta_p[0, 1]
            c += 0.00001 * delta_p[0, 2]
            d += 0.00001 * delta_p[0, 3]
            e += 0.00001 * delta_p[0, 4]
            f += 0.00001 * delta_p[0, 5]
            # 更新欧拉角和平移向量的值，学习率分别为0.00005,0.01
            for i in range(N):
                # 欧拉角
                self.alpha[i] += 0.000001 * delta_p[0, 6 + 6 * i]
                self.beta[i] += 0.000001 * delta_p[0, 6 + 6 * i + 1]
                self.gamma[i] += 0.000001 * delta_p[0, 6 + 6 * i + 2]
                # 平移向量
                self.T1[i] += 0.00001 * delta_p[0, 6 + 6 * i + 3]
                self.T2[i] += 0.00001 * delta_p[0, 6 + 6 * i + 4]
                self.T3[i] += 0.00001 * delta_p[0, 6 + 6 * i + 5]

            # 计算RMSE指标数值，判断是否需要下轮更新。
            self.R = []
            self.t = []
            for i in range(N):
                # 用优化后的欧拉角转换成旋转矩阵
                R = eulerAnglesToRotationMatrix(self.alpha[i], self.beta[i], self.gamma[i])
                self.R.append(R)
                self.t.append(self.T1[i])
                self.t.append(self.T2[i])
                self.t.append(self.T3[i])

            self.R = np.array(self.R).reshape(N * 3, 3)
            self.t = np.array(self.t)
            temp_m = np.array([0, a, b, d, 0, 0, c, e, 0, 0, 0, f]).reshape(3, 4)
            delta_M = self.R @ temp_m + self.t
            RMSE = RMSELoss(delta_M, P_test)  # 参数1预测值，参数2真实值
            epoch = epoch + 1
            print('RMSE:%f' % RMSE)

    def jacobian_matrix(self):
        """
        description：
            利用torch包计算梯度下降优化中的雅可比矩阵
        :param: function: loss函数
        :param x1: PE1-模板坐标系对应的第一个小球坐标
        :param x2: PE2-模板坐标系对应的第二个小球坐标
        :param x3: PE3-模板坐标系对应的第三个小球坐标
        :param x4: PE4-模板坐标系对应的第四个小球坐标
        """

        # 将参数转换成tensor类型
        a = self.Pt_1[0].copy()
        b = self.Pt_2[0].copy()
        c = self.Pt_2[1].copy()
        d = self.Pt_3[0].copy()
        e = self.Pt_3[1].copy()
        f = self.Pt_3[2].copy()

        a = torch.tensor(a)
        b = torch.tensor(b)
        c = torch.tensor(c)
        d = torch.tensor(d)
        e = torch.tensor(e)
        f = torch.tensor(f)
        # self.R = torch.tensor(self.R)
        # self.t = torch.tensor(self.t)
        self.alpha = torch.tensor(self.alpha)
        self.beta = torch.tensor(self.beta)
        self.gamma = torch.tensor(self.gamma)
        self.T1 = torch.tensor(self.T1)
        self.T2 = torch.tensor(self.T2)
        self.T3 = torch.tensor(self.T3)

        # J_torch = jacobian(func, (x1, x2, x3, x4))
        J_torch = jacobian(self.cost_function,
                           (a, b, c, d, e, f, self.alpha, self.beta, self.gamma, self.T1, self.T2, self.T3))

        dn1 = len(J_torch)
        dn2 = len(J_torch[1])

        # J_np = np.zeros((dn1, dn2 * 9))
        # # 将torch计算后得到的雅可比矩阵转换成numpy格式(shape(4,36)),并从shape(4,36)->(12,12)
        # for i in range(len(J_torch)):
        #     for j in range(len(J_torch[i])):
        #         temp = J_torch[j][i]
        #         temp = temp.numpy()
        #         temp = temp.reshape(1, 9)
        #
        #         J_np[i][j * 9] = temp[0][0]
        #         J_np[i][j * 9 + 1] = temp[0][1]
        #         J_np[i][j * 9 + 2] = temp[0][2]
        #         J_np[i][j * 9 + 3] = temp[0][3]
        #         J_np[i][j * 9 + 4] = temp[0][4]
        #         J_np[i][j * 9 + 5] = temp[0][5]
        #         J_np[i][j * 9 + 6] = temp[0][6]
        #         J_np[i][j * 9 + 7] = temp[0][7]
        #         J_np[i][j * 9 + 8] = temp[0][8]
        #
        # # 重新建立一个矩阵按照需要的格式填入J_np里的数据
        # # shape(12,12)
        # J = np.zeros((12, 12))
        # for j in range(4):
        #     # for i in range(3):
        #     J[j * 3][j * 3] = J_np[j][j * 9]
        #     J[j * 3][j * 3 + 1] = J_np[j][j * 9 + 1]
        #     J[j * 3][j * 3 + 2] = J_np[j][j * 9 + 2]
        #     J[j * 3 + 1][j * 3] = J_np[j][j * 9 + 3]
        #     J[j * 3 + 1][j * 3 + 1] = J_np[j][j * 9 + 4]
        #     J[j * 3 + 1][j * 3 + 2] = J_np[j][j * 9 + 5]
        #     J[j * 3 + 2][j * 3] = J_np[j][j * 9 + 6]
        #     J[j * 3 + 2][j * 3 + 1] = J_np[j][j * 9 + 7]
        #     J[j * 3 + 2][j * 3 + 2] = J_np[j][j * 9 + 8]
        # # 将之前的tensor类型转换回numpy类型
        # self.R = self.R.numpy()
        # self.t = self.t.numpy()

        return J_torch


"""
尝试使用SGD优化器，对参数进行优化：
"""


def model(a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3, N):
    """
    对于估计值的建模
    :param N:
    :param a: 参数a
    :param b:
    :param c:
    :param d:
    :param e:
    :param f:
    :param alpha:
    :param beta:
    :param gamma:
    :param T1:
    :param T2:
    :param T3:
    :return:
    """

    E = torch.zeros(N * 12, 1)

    for i in range(N):
        E[i * 12] = T1[i]
        E[i * 12 + 1] = T2[i]
        E[i * 12 + 2] = T3[i]

        E[i * 12 + 3] = T1[i] + a * math.cos(alpha[i]) * math.cos(beta[i])
        E[i * 12 + 4] = T2[i] - a * (
                math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
            gamma[i]))
        E[i * 12 + 5] = T3[i] + a * (
                math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
            beta[i]))

        E[i * 12 + 6] = T1[i] + b * math.cos(alpha[i]) * math.cos(beta[i]) + c * math.cos(
            beta[i]) * math.sin(alpha[i])
        E[i * 12 + 7] = T2[i] - b * (
                math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(beta[i]) * math.sin(beta[i]) * math.sin(
            gamma[i])) + c * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
            beta[i]) * math.sin(gamma[i]))
        E[i * 12 + 8] = T3[i] + b * (
                math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
            beta[i])) - c * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
            alpha[i]) * math.sin(beta[i]))

        E[i * 12 + 9] = T1[i] - f * math.sin(beta[i]) + d * math.cos(alpha[i]) * math.cos(
            beta[i]) + e * math.cos(beta[i]) * math.sin(alpha[i])
        E[i * 12 + 10] = T2[i] - d * (
                math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
            gamma[i])) + e * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
            beta[i]) * math.sin(gamma[i])) + f * math.cos(beta[i]) * math.sin(gamma[i])
        E[i * 12 + 11] = T3[i] + d * (
                math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
            beta[i])) - e * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
            alpha[i]) * math.sin(beta[i])) + f * math.cos(beta[i]) * math.cos(gamma[i])

    # print(E)
    return E


def loss_fn(y_pred, y_test):
    """
    计算rmse

    :param y_pred:预测值
    :param y_test:实测值
    :return:RMSE
    """
    MSE = torch.sum((y_test - y_pred) ** 2) / len(y_test)
    RMSE = torch.sqrt(MSE)
    return RMSE


def RMSELoss(yhat, y):
    """

    :param yhat: 预测值
    :param y: 真实值
    :return:
    """
    return np.sqrt(np.mean((yhat - y) ** 2))


def train_loop(n_epochs, optimizer0, optimizer1, a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3, N, y_test):
    """

    :param n_epochs:
    :param optimizer:
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :param f:
    :param alpha:
    :param beta:
    :param gamma:
    :param T1:
    :param T2:
    :param T3:
    :param N:
    :param y_test:
    :return:
    """
    for epoch in range(n_epochs):
        y_pred = model(a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3, N)
        RMSE = loss_fn(y_pred, y_test)

        optimizer0.zero_grad()
        optimizer1.zero_grad()
        RMSE.backward()

        # 参数的值会在调用step后更新
        optimizer0.step()
        optimizer1.step()

        if epoch % 100 == 0:
            print('Epoch %d ,RMSE %f' % (epoch, float(RMSE)))

    return a, b, c, d, e, f


if __name__ == '__main__':
    # 获取当前脚本所在文件夹的路径
    curpath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路经
    # Yaml_name = input("please Fill in the name of the .yaml file (path): ")
    yamlpath = os.path.join(curpath, "../YamlFiles/experience_data2.yaml")
    # yamlpath = os.path.join(curpath, Yaml_name)
    yaml_op1 = yaml_handle(yamlpath)
    data = yaml_op1.get_yaml()
    Data = yaml_op1.conver_yaml(data)

    # N:数据维度
    N = len(data)
    # 将数据按（列）重新reshape,"f":按照列填入
    Data = Data.reshape(Data.shape[0], 3, 4, order='F')
    print(Data.shape)
    # 创建实例对象
    template_data = template(Data)
    template_data.Template_PointReorder()
    # 创建初始化模板
    template_init = template_data.Template_initBuild(0)
    # 梯度下降优化模板
    template_data.theta_Dataproc()
    template_data.Template_OPT()
    """
    梯度下降优化部分：
    优化器失败了，暂时做不出来 2023/5/4
    """
    # # 待优化参数提取
    # a, b, c, d, e, f, alpha, beta, gamma, T1, T2, T3, P_M = template_data.theta_Dataproc()
    # # 将待优化参数转换成张量，并允许计算梯度
    # a = torch.tensor(a, requires_grad=True)
    # b = torch.tensor(b, requires_grad=True)
    # c = torch.tensor(c, requires_grad=True)
    # d = torch.tensor(d, requires_grad=True)
    # e = torch.tensor(e, requires_grad=True)
    # f = torch.tensor(f, requires_grad=True)
    # alpha = torch.tensor(alpha, requires_grad=True)
    # beta = torch.tensor(beta, requires_grad=True)
    # gamma = torch.tensor(gamma, requires_grad=True)
    # T1 = torch.tensor(T1, requires_grad=True)
    # T2 = torch.tensor(T2, requires_grad=True)
    # T3 = torch.tensor(T3, requires_grad=True)
    # P_M = torch.tensor(P_M, requires_grad=True)
    # # 建立并初始化优化器,将动量设置为0就是简单的批处理梯度下降
    # # learning_rate = 1e-1
    # optimizer0 = optim.Adam([a, b, c, d, e, f, T1, T2, T3], lr=1e-1)
    # optimizer1 = optim.Adam([alpha, beta, gamma], lr=1e-5)
    # # 迭代训练
    # a1, b1, c1, d1, e1, f1 = train_loop(n_epochs=3000, optimizer0=optimizer0, optimizer1=optimizer1, a=a, b=b, c=c, d=d,
    #                                     e=e, f=f, alpha=alpha,
    #                                     beta=beta, gamma=gamma, T1=T1, T2=T2, T3=T3, N=N, y_test=P_M)
    # print(a1, b1, c1, d1, e1, f1)
