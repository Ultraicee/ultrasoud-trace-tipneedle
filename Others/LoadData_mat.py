import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as io
import yaml


def reshapeData(data_name, Variable_name):
    """
    :param data_name(.mat文件的名字，.mat文件建立的时候需要命名矩阵为'Tracedata')
            Variable_name(mat文件中的数据变量名字）
    将采集到的三维图像数据进行有规律的编排
    :return:
    排列好的数据Data1
    """

    data = io.loadmat(data_name)
    Data = np.array(data[Variable_name])
    rows = Data.shape[0]  # 原数据的行
    cols = Data.shape[1]  # 原数据的列
    print("Source Data's shape:", Data.shape)

    x = Data[:, 0]
    y = Data[:, 1]
    z = Data[:, 2]

    Data1 = np.zeros(cols * rows)
    Data1 = Data1.reshape(int(rows / 4), 1, 12)
    for i in range(0, int(rows / 4)):
        Data1[i, :, :] = [x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3],
                          y[i * 4], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3],
                          z[i * 4], z[i * 4 + 1], z[i * 4 + 2], z[i * 4 + 3]]
    Data1 = Data1.reshape(int(rows / 4), 3, 4)
    print("reshape size ", Data1.shape)
    return Data1


def reshapeData_ultrasoundimg(data_name, Variable_name):
    """
    :param data_name(.mat文件的名字，.mat文件建立的时候需要命名矩阵为'Tracedata')
            Variable_name(mat文件中的数据变量名字）
    将采集到的超声波二维图像的原数据增加一维，并进行(Nx3x6)的编排。
    :return:
    排列好的数据Data1
    """

    data = io.loadmat(data_name)
    Data = np.array(data[Variable_name])
    # data = io.loadmat('Tracedata.mat')
    # Data = np.array(data['Tracedata'])
    rows = Data.shape[0]  # 原数据的行
    cols = Data.shape[1]  # 原数据的列
    print("Source Data's shape:", Data.shape)

    x = Data[:, 0]
    y = Data[:, 1]
    z = numpy.zeros(rows)

    Data1 = np.zeros(cols * rows + np.size(z))
    Data1 = Data1.reshape(int(rows / 6), 1, 18)
    for i in range(0, int(rows / 6)):
        Data1[i, :, :] = [x[i * 6], x[i * 6 + 1], x[i * 6 + 2], x[i * 6 + 3], x[i * 6 + 4], x[i * 6 + 5],
                          y[i * 6], y[i * 6 + 1], y[i * 6 + 2], y[i * 6 + 3], y[i * 6 + 4], y[i * 6 + 5],
                          z[i * 6], z[i * 6 + 1], z[i * 6 + 2], z[i * 6 + 3], z[i * 6 + 4], z[i * 6 + 5]]
    Data1 = Data1.reshape(int(rows / 6), 3, 6)
    print("reshape size ", Data1.shape)
    return Data1


def P3dshow(Data1):
    """
    将重新排列好的手术器械小球三维重建位置
    :param Data1:
    :return:
    """
    rows = Data1.shape[0]  # 原数据的行
    cols = Data1.shape[1]  # 原数据的列
    xd = Data1[0, 0, 0]
    yd = Data1[0, 1, 0]
    zd = Data1[0, 2, 0]

    ax = plt.subplot(111, projection='3d')
    for i in range(0, int(rows)):
        xd = Data1[i, 0, :]
        yd = Data1[i, 1, :]
        zd = Data1[i, 2, :]

        ax.plot3D(xd, yd, zd, 'gray', linewidth='1')
        ax.scatter(xd, yd, zd, c='r', s=2)

    xd1 = -48.17276991
    yd1 = -23.53846245
    zd1 = 250.51769171
    ax.plot3D(xd1, yd1, zd1, 'r', linewidth='1')

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()
