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
    # yamlpath = os.path.join(curpath, "../YamlFiles/experience_data1.yaml")
    yamlpath = os.path.join(curpath, Yaml_name)
    yaml_op1 = yaml_create.yaml_handle(yamlpath)
    data = yaml_op1.get_yaml()
    Data = yaml_op1.conver_yaml(data)

    # N:数据的大小
    N = len(data)
    # print(len(data))
    # 计算针尖在世界坐标系和器械坐标系下的位置
    CalNeedleTip.calneedletip(Data)


