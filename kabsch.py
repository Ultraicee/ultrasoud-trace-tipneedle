import numpy as np


def kabsch(p, q):
    # P和q是两个点集的坐标。
    # p和q应该是尺寸（N，3）的二维数组，其中N是点数。
    # 每个数组的3列应分别包含每个点的x、y和z坐标。

    # 计算两个点集的质心。
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    # 通过减去它们的质心来使点集居中。
    p_centered = p - centroid_p
    q_centered = q - centroid_q

    # 计算中心点集的协方差矩阵。
    cov = p_centered.T.dot(q_centered)

    # 计算协方差矩阵的奇异值分解。
    U, S, V = np.linalg.svd(cov)

    # 通过取U和V矩阵的点积来计算旋转矩阵。
    R = U.dot(V)

    # 通过取质心的差异来计算平移矢量
    # 两个点集，并将其乘以旋转矩阵。
    T = centroid_p - R.dot(centroid_q)

    return R, T



# q = np.array([[-55.609428, -76.902779, -36.621086, -53.637581],
#               [-22.828926, -82.217972, -62.976185, 7.1691613],
#               [415.49362, 383.93573, 386.25583, 433.96109]])
#
# p = np.transpose(p)
# q = np.transpose(q)
# print(p.shape)
# R, T = kabsch(p, q)
# print(R,T)
