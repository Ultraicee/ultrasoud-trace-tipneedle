import numpy as np
import scipy
from Others import build_template
# 定义4个三维点的坐标
p1 = [-292.22600, 82.09300, 940.72300]
p2 = [-314.62900, 120.36300, 948.17100]
p3 = [-247.89000, 108.34900, 929.61000]
p4 = [-214.04100, 111.74300, 919.31700]

# 将4个点的坐标转换为numpy数组
points = np.array([p1, p2, p3, p4])


def cal(a, b, c, d):
    a1 = scipy.linalg.norm(abs(a - center))
    b1 = scipy.linalg.norm(abs(b - center))
    c1 = scipy.linalg.norm(abs(c - center))
    d1 = scipy.linalg.norm(abs(d - center))
    print(a1, b1, c1, d1)
    value = min(a1, b1, c1, d1)
    print(value)


# 计算四个点的中心点
center = np.mean(points, axis=0)

print(center)
cal(p1, p2, p3, p4)

if __name__ == '__main__':
    test = build_template.template()