import numpy as np
import math
import scipy
import torch
from Others.kabsch import kabsch
# 尝试利用pytorch方法解决雅可比矩阵问题
from torch.autograd.functional import jacobian


class template:
    """
    description:
        建造针尖模板坐标系。
    """

    def __init__(self, measure3d, degree=1, lr=0.01, num_iter=1000):
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
        self.reorder_P0 = 0
        self.reorder_P1 = 0
        self.reorder_P2 = 0
        self.reorder_P3 = 0

        # 模板坐标系Pt各点坐标
        self.Pt_0 = 0
        self.Pt_1 = 0
        self.Pt_2 = 0
        self.Pt_3 = 0

        # 制作模板时候使用的flag
        self.p_flag = [0, 0, 0, 0]

        # RT旋转平移矩阵temp
        self.R = 0
        self.t = 0

        # 梯度下降相关函数
        self.degree = degree
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None

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

        :return: no
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

        print("Template has ordered.")

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
            temp
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

    def isRotationMatrix(self, R):
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

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):
        """
        description:
            将旋转矩阵转换成欧拉角
        :param R: 旋转矩阵
        :return:角度值 - x,y,z
        """
        assert (self.isRotationMatrix(R))

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

    def eulerAnglesToRotationMatrix(self, theta):
        """
        description:
            将欧拉角转换成旋转矩阵
        :param: theta: [x,y,z]
        :return: R: shape(3,3) 旋转矩阵
        """
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def Matrix_RT(self, N):
        """
        description:
            计算初始模板坐标系和测量数据的平移旋转矩阵
        :param N: 第N组测量数据
        :return:NONE
        """
        Measure = np.array([self.reorder_P0[N], self.reorder_P1[N], self.reorder_P2[N], self.reorder_P3[N]])
        template_N = np.array([self.Pt_0, self.Pt_1, self.Pt_2, self.Pt_3])
        self.R, self.t = kabsch(Measure, template_N)

    def cost_function(self, x1, x2, x3, x4):
        """
        description:
            损失函数本体表达
        :param x1: PE1-模板坐标系对应的第一个小球坐标
        :param x2: PE2-模板坐标系对应的第二个小球坐标
        :param x3: PE3-模板坐标系对应的第三个小球坐标
        :param x4: PE4-模板坐标系对应的第四个小球坐标

        :return: type:tuple
        """

        return self.R @ x1 + self.t, self.R @ x2 + self.t, self.R @ x3 + self.t, self.R @ x4 + self.t

    def jacobian_matrix(self, func, x1, x2, x3, x4):
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
        x1 = torch.tensor(x1)
        x2 = torch.tensor(x2)
        x3 = torch.tensor(x3)
        x4 = torch.tensor(x4)
        self.R = torch.tensor(self.R)
        self.t = torch.tensor(self.t)
        J_torch = jacobian(func, (x1, x2, x3, x4))
        dn1 = len(J_torch)
        dn2 = len(J_torch[1])

        J_np = np.zeros((dn1, dn2 * 9))
        # 将torch计算后得到的雅可比矩阵转换成numpy格式(shape(4,36)),并从shape(4,36)->(12,12)
        for i in range(len(J_torch)):
            for j in range(len(J_torch[i])):
                temp = J_torch[j][i]
                temp = temp.numpy()
                temp = temp.reshape(1, 9)

                J_np[i][j * 9] = temp[0][0]
                J_np[i][j * 9 + 1] = temp[0][1]
                J_np[i][j * 9 + 2] = temp[0][2]
                J_np[i][j * 9 + 3] = temp[0][3]
                J_np[i][j * 9 + 4] = temp[0][4]
                J_np[i][j * 9 + 5] = temp[0][5]
                J_np[i][j * 9 + 6] = temp[0][6]
                J_np[i][j * 9 + 7] = temp[0][7]
                J_np[i][j * 9 + 8] = temp[0][8]

        # 重新建立一个矩阵按照需要的格式填入J_np里的数据
        # shape(12,12)
        J = np.zeros((12, 12))
        for j in range(4):
            # for i in range(3):
            J[j * 3][j * 3] = J_np[j][j * 9]
            J[j * 3][j * 3 + 1] = J_np[j][j * 9 + 1]
            J[j * 3][j * 3 + 2] = J_np[j][j * 9 + 2]
            J[j * 3 + 1][j * 3] = J_np[j][j * 9 + 3]
            J[j * 3 + 1][j * 3 + 1] = J_np[j][j * 9 + 4]
            J[j * 3 + 1][j * 3 + 2] = J_np[j][j * 9 + 5]
            J[j * 3 + 2][j * 3] = J_np[j][j * 9 + 6]
            J[j * 3 + 2][j * 3 + 1] = J_np[j][j * 9 + 7]
            J[j * 3 + 2][j * 3 + 2] = J_np[j][j * 9 + 8]
        # 将之前的tensor类型转换回numpy类型
        self.R = self.R.numpy()
        self.t = self.t.numpy()

        return J

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
            self.Matrix_RT(i)
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
            self.Matrix_RT(i)
            loss.append(self.cost_function(self.Pt_0, self.Pt_1, self.Pt_2, self.Pt_3))
        print(loss)
