import numpy as np
import math
import scipy
import torch
from Others.kabsch import kabsch
# 尝试利用pytorch方法解决雅可比矩阵问题
from torch.autograd.functional import jacobian


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
        c = self.calp2line(self.reorder_P1[N], self.reorder_P0[N], self.reorder_P2[N])
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

        :return:
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
        # 将整理好的所需数据塞入私有
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

        return alpha, beta, gamma, T1, T2, T3

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
        # Pe1 = np.zeros((3, 1))
        # Pe2 = np.zeros((3, 1))
        # Pe3 = np.zeros((3, 1))
        # Pe4 = np.zeros((3, 1))

        # 因为torch的拼接问题，先将3个平移向量重新转回numpy的格式
        # T1 = T1.detach().numpy()
        # T2 = T2.detach().numpy()
        # T3 = T3.detach().numpy()

        # Pe2[0][0] = a
        # Pe3[0][0] = b
        # Pe3[1][0] = c
        # Pe4[0][0] = d
        # Pe4[1][0] = e
        # Pe4[2][0] = f
        # print(Pe1, Pe2, Pe3, Pe4)
        # 获取数据的大小
        N = self.Fig_N

        # 将欧拉角转换成旋转矩阵
        # for i in range(N):
        #     R_temp = eulerAnglesToRotationMatrix(alpha[i], beta[i], gamma[i])
        #     self.R.append(R_temp)
        #     self.t.append(T1[i])
        #     self.t.append(T2[i])
        #     self.t.append(T3[i])
        # self.R = np.array(self.R).reshape(3 * N, 3)
        # self.t = np.array(self.t)
        # self.t = torch.cat(self.t,dim=0)

        p0 = self.reorder_P0.copy().reshape(3 * N, 1)
        p1 = self.reorder_P1.copy().reshape(3 * N, 1)
        p2 = self.reorder_P2.copy().reshape(3 * N, 1)
        p3 = self.reorder_P3.copy().reshape(3 * N, 1)

        # self.R = torch.tensor(self.R)
        # self.t = torch.tensor(self.t)
        # Pe1 = torch.tensor(Pe1)
        # Pe2 = torch.tensor(Pe2)
        # Pe3 = torch.tensor(Pe3)
        # Pe4 = torch.tensor(Pe4)
        p0 = torch.tensor(p0)
        p1 = torch.tensor(p1)
        p2 = torch.tensor(p2)
        p3 = torch.tensor(p3)

        # E_1 = self.R @ Pe1 + self.t - p0
        # E_2 = self.R @ Pe2 + self.t - p1
        # E_3 = self.R @ Pe3 + self.t - p2
        # E_4 = self.R @ Pe4 + self.t - p3

        E = torch.zeros(N * 12, 1)

        for i in range(N):
            E[i * 12] = self.T1[i] - p0[i * 3]
            E[i * 12 + 1] = self.T2[i] - p0[i * 3 + 1]
            E[i * 12 + 2] = self.T3[i] - p0[i * 3 + 2]

            E[i * 12 + 3] = self.T1[i] - p1[i * 3] + a * math.cos(alpha[i]) * math.cos(beta[i])
            E[i * 12 + 4] = self.T2[i] - p1[i * 3 + 1] - a * (
                    math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
                gamma[i]))
            E[i * 12 + 5] = self.T3[i] - p1[i * 3 + 2] + a * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i]))

            E[i * 12 + 6] = self.T1[i] - p2[i * 3] + b * math.cos(alpha[i]) * math.cos(beta[i]) + c * math.cos(
                beta[i]) * math.sin(alpha[i])
            E[i * 12 + 7] = self.T2[i] - p2[i * 3 + 1] - b * (
                    math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(beta[i]) * math.sin(beta[i]) * math.sin(
                gamma[i])) + c * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i]))
            E[i * 12 + 8] = self.T3[i] - p2[i * 3 + 2] + b * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i])) - c * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
                alpha[i]) * math.sin(beta[i]))

            E[i * 12 + 9] = self.T1[i] - p3[i * 3] - f * math.sin(beta[i]) + d * math.cos(alpha[i]) * math.cos(
                beta[i]) + e * math.cos(beta[i]) * math.sin(alpha[i])
            E[i * 12 + 10] = self.T2[i] - p3[i * 3 + 1] - d * (
                        math.cos(gamma[i]) * math.sin(alpha[i]) - math.cos(alpha[i]) * math.sin(beta[i]) * math.sin(
                    gamma[i])) + e * (math.cos(alpha[i]) * math.cos(gamma[i]) + math.sin(alpha[i]) * math.sin(
                beta[i]) * math.sin(gamma[i])) + f * math.cos(beta[i]) * math.sin(gamma[i])
            E[i * 12 + 11] = self.T3[i] - p3[i * 3 + 2] + d * (
                    math.sin(alpha[i]) * math.sin(gamma[i]) + math.cos(alpha[i]) * math.cos(gamma[i]) * math.sin(
                beta[i])) - e * (math.cos(alpha[i]) * math.sin(gamma[i]) - math.cos(gamma[i]) * math.sin(
                alpha[i]) * math.sin(beta[i])) + f * math.cos(beta[i]) * math.cos(gamma[i])

        # loss = np.linalg.norm(E)
        # loss = torch.tensor(loss)

        # print(E)
        return E

    # return self.R @ x1 + self.t, self.R @ x2 + self.t, self.R @ x3 + self.t, self.R @ x4 + self.t

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

    def Template_opt(self, alpha, epoch, x1, x2, x3, x4):
        """
        description:
        梯度下降主函数
        :param: alpha 步长
        :param: epoch 遍历次数
        :return:包含参数[a,b,c,d,e,f]最优参数的矩阵
        """

        theta = 1000
        Jacobian = self.jacobian_matrix(self.cost_function, x1, x2, x3, x4)
        E_theta = []
        for i in range(epoch):
            self.Matrix_RT_Conversion(i)
            loss = self.cost_function(x1, x2, x3, x4)  # 得到元组构成的loss

            for j in range(len(loss)):
                E_theta.append(loss[j])
            E_theta = np.array(E_theta)
            E_theta = E_theta.reshape(1, 12)
            theta = theta - alpha * E_theta @ Jacobian

        return theta

    def Template_total(self):
        """
        description:
            计算并优化模板坐标系的总执行函数。
        :param:
        :return:
        """
        j = []
        loss = []
        for i in range(1, self.Fig_N):
            self.Matrix_RT_Conversion(i)
            loss.append(self.cost_function(self.Pt_0, self.Pt_1, self.Pt_2, self.Pt_3))
        print(loss)
