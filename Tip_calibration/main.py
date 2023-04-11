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
    print(Data.shape)
    # 计算针尖在世界坐标系和器械坐标系下的位置
    CalNeedleTip.calneedletip(Data)

    # 按列重新reshape,"f":按照列填入
    Data = Data.reshape(Data.shape[0], 3, 4, order='F')
    print(Data.shape)
    # LoadData_mat.P3dshow(Data)

    test = build_template.template(Data)
    value = test.reorder()
    tamplate = test.Template_build(0)

    # tamplate = tamplate.T
    # test_data = np.array([test.reorder_P0[0], test.reorder_P1[0], test.reorder_P2[0], test.reorder_P3[0]])
    #
    # R, t = kabsch.kabsch(test_data, tamplate)
    # print(R)
    # print(t)
    # print("-----------------")
    # new0 = R @ tamplate[0] + t
    # new1 = R @ tamplate[1] + t
    # new2 = R @ tamplate[2] + t
    # new3 = R @ tamplate[3] + t
    # new = np.array([new0, new1, new2, new3])
    # loss = new - test_data
    # print(loss)

    print(test.loss_function(tamplate, 0))
