import torch

from Others import LoadData_mat, yaml_create, build_template, kabsch
from Tip_calibration import CalNeedleTip
import numpy as np
import os

if __name__ == '__main__':
    # 获取当前脚本所在文件夹的路经
    curpath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路经
    Yaml_name = input("please Fill in the name of the .yaml file (path): ")
    # yamlpath = os.path.join(curpath, "YamlFiles/experience_data1.yaml")
    yamlpath = os.path.join(curpath, Yaml_name)
    yaml_op1 = yaml_create.yaml_handle(yamlpath)
    data = yaml_op1.get_yaml()
    Data = yaml_op1.conver_yaml(data)

    N = len(data)
    # print(len(data))
    # 计算针尖在世界坐标系和器械坐标系下的位置
    CalNeedleTip.calneedletip(Data)

    # 按列重新reshape,"f":按照列填入
    Data = Data.reshape(Data.shape[0], 3, 4, order='F')
    print(Data.shape)
    # LoadData_mat.P3dshow(Data)

    test = build_template.template(Data)
    test.Template_PointReorder()
    template = test.Template_initBuild(0)  # 使用第一帧进行模板坐标系的建立，后面将不再使用第一帧。

    """
        求模板坐标集合，看看后面能否使用最小二乘求解
        :不行，误差太大了
    """
    # template = np.zeros((N, 12))
    # for i in range(N):
    #     temp = test.Template_build(i).reshape(1, 12)
    #     template[i] = temp
    # print("模板坐标集合为: ", template)

    """
        求loss函数的雅可比矩阵
    """
    pe1 = template.T[0]
    pe2 = template.T[1]
    pe3 = template.T[2]
    pe4 = template.T[3]
    # loss = test.loss_function(pe1, pe2, pe3, pe4)
    # print(loss)
    test.Matrix_RT(1)
    f = test.loss_function(pe1, pe2, pe3, pe4)
    # J = test.jacobian_matrix(test.loss_function, pe1, pe2, pe3, pe4)
    t_value = test.Template_opt(0.01, 1, pe1, pe2, pe3, pe4)
    print(t_value)
