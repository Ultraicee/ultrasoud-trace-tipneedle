import numpy as np
from Others.kabsch import kabsch

# 输入优化好的器械坐标系
p_tmep = np.array([[0., 7.0747100000000003e+01, 4.1330100000000002e+01, -3.4146000000000001e+01],
                   [0., 0., 3.3898200000000003e+01, -9.1819199999999999e+00],
                   [0., 0., 0., 3.1964500000000001e-01]])


def calneedletip(Data):
    dn = Data.shape[0]  # dn组数据
    Data = Data.reshape(dn, 3, 4)
    R = np.zeros(dn * 3 * 3).reshape(dn, 3, 3)
    Rm = np.zeros(dn * 3 * 6).reshape(dn, 3, 6)
    T = np.zeros(dn * 1 * 3).reshape(dn, 1, 3)
    # 将器械坐标系和采集到N组小球数据进行kabsch计算，得到N组旋转平移坐标
    I = np.identity(3)
    for i in range(0, dn):
        R[i, :, :], T[i, :, :] = kabsch(np.transpose(Data[i, :, :]), np.transpose(p_tmep))  #
        Rm[i, :, :] = np.hstack((R[i, :, :], -I))
    # T = T.reshape(dn, 3, 1)
    # print(R.shape)
    #   假设针尖的坐标为needletip_m,根据公式 (R_i-R_j)*needletip_m = -(T_i-T_j)
    #   可以求出大体针尖的坐标位置
    # Rm = Ri-Rj,Ri为每组的数据，令Rj为单位矩阵，每次进行计算

    # 最小二乘法求解针尖
    T = T.reshape(dn * 3, 1)
    Rm = Rm.reshape(dn * 3, 6)
    needletip_cal = np.linalg.inv(np.transpose(Rm) @ Rm) @ np.transpose(Rm) @ (-T)
    needletip_D = needletip_cal[:3, :]
    needletip_m = needletip_cal[3:6, :]
    # print(needletip_cal)
    print("器械坐标下的针尖坐标为：", needletip_D.T)
    print("相机坐标下的针尖坐标为：", needletip_m.T)

# data = LoadData.reshapeData()
# calneedletip(data)
