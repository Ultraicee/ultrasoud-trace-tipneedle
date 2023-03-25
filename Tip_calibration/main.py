from Others import LoadData_mat, yaml_create,build_template
from Tip_calibration import CalNeedleTip
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
    value = test.Template_build()
    print(value)


