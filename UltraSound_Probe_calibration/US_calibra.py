import math

import scipy.optimize as opt
import numpy as np
from Others import kabsch, yaml_create
from Others.build_template import eulerAnglesToRotationMatrix
from scipy.spatial.transform import Rotation

# p1,p2为利用针尖标定的N线端点三维坐标，probe为探头坐标系下的探头坐标
p1 = np.array([[-47.7666, -4.6951, 0.3865, -44.6705, -1.4557, 2.9381],
               [118.5508, 123.9115, 124.4499, 188.0993, 192.6857, 193.9377],
               [633.4253, 619.2024, 617.3193, 671.5347, 657.7082, 655.6947]])

p2 = np.array([[-8.7369, -3.0507, 39.9845, -3.7306, 1.4531, 43.9121],
               [67.4820, 61.2111, 66.0803, 131.6405, 132.7506, 138.6730],
               [742.9518, 736.3498, 722.5941, 775.2590, 774.5312, 760.4695]])

probe = np.array([[5.2090, 70, -5.2090, -30], [-29.5440, 0, 29.5440, 0], [0, 0, 0, 0]])


def f(para):
    # 从yaml文件读取超声波探头坐标数据
    yamlpath = "../YamlFiles/ultrasound_probe_data.yaml"
    yaml_op1 = yaml_create.yaml_handle(yamlpath)
    data1 = yaml_op1.get_yaml()
    US_probe_data = yaml_op1.conver_yaml(data1, 'probecam')
    US_probe_data = US_probe_data.reshape(US_probe_data.shape[0], 3, 4)

    # 从yaml文件读取超声图像标记点数据
    yamlpath = "../YamlFiles/ultrasound_pixel_data.yaml"
    yaml_op2 = yaml_create.yaml_handle(yamlpath)
    data2 = yaml_op2.get_yaml()
    US_uv_data = yaml_op2.conver_yaml(data2, 'ultrasound-pixel')
    US_uv_data = US_uv_data.reshape(US_uv_data.shape[0], 2, 6)

    # 导入初始值
    R3 = eulerAnglesToRotationMatrix(para[0], para[1], para[2])
    # R3 = Rotation.from_euler('zyx', [para[0], para[1], para[2]], degrees=True).as_matrix()
    # print(R3)
    t3 = np.array([para[3], para[4], para[5]])
    # 比例因子
    sx = para[6]
    sy = para[7]
    # 初始误差方程
    errMatrix = US_uv_data

    # 优化循环
    for m in range(US_uv_data.shape[0]):
        # 求取每帧摄像机坐标下小球坐标和探头的坐标位置变换矩阵和平移量
        R2, t2 = kabsch.kabsch(probe.T, US_probe_data[m, :, :].T)
        # R2 = Rotation.align_vectors(US_probe_data[m, :, :].T, probe.T)

        # 此Rt表示从超声图像直接变换到摄像机坐标系的变换矩阵和平移量
        R = R3 @ R2
        t = R3 @ t2 + t3
        """
        根据直线方程表示：
        X_r-X_1     Y_r-Y_1     Z_r-Z_1
        -------- = --------- = ---------
        X_2-X1      Y_2-Y_1     Z_2-Z_1
        """

        Yr = -(t[2] * (p2[1, :] - p1[1, :]) + R[2, 0] * (p2[1, :] * p1[0, :] - p1[1, :] * p2[0, :]) + R[2, 2] * (
                p2[1, :] * p1[2, :] - p1[1, :] * p2[2, :])) \
             / (R[2, 0] * (p2[0, :] - p1[0, :]) + R[2, 1] * (p2[1, :] - p1[1, :]) + R[2, 2] * (p2[2, :] - p1[2, :]))

        Xr = ((p2[0, :] - p1[0, :]) * Yr - p1[1, :] * p2[0, :] + p2[1, :] * p1[0, :]) / (p2[1, :] - p1[1, :])

        Zr = ((p2[2, :] - p1[2, :]) * Yr - p1[1, :] * p2[2, :] + p2[1, :] * p1[2, :]) / (p2[1, :] - p1[1, :])

        # 当前帧的超声图像标记点二维坐标
        uv = US_uv_data[m, :, :]

        # 将Rt拼接在一起并转换为齐次矩阵
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        # 将Xr,Yr,Zr拼接在一起，构成一个齐次方程与上面的旋转平移矩阵相乘
        P = np.ones((4, 6))
        P[0, :] = Xr
        P[1, :] = Yr
        P[2, :] = Zr

        # 计算误差，公式原型参考刘海金论文公式（3-43）
        errMatrix[m, :, :] = H[:2, :] @ P - (np.array([[sx], [sy]]) * uv)

    err = errMatrix.reshape((1, errMatrix.shape[0] * errMatrix.shape[1] * errMatrix.shape[2]))
    rmse = math.sqrt(np.sum(err ** 2) / 90)
    print(rmse)
    return rmse


if __name__ == '__main__':
    # 输入待优化的八个初始值：alpha,beta,grammar,tx,ty,tz,sx,sy
    para0 = np.array([0.8, 0, 0, 130, -90, -42, 0.117, 0.116])
    # 求解非线性方程解
    para_opt = opt.minimize(f, para0).x
    print(para_opt)
