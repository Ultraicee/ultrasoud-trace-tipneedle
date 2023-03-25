import yaml
import os
import numpy as np
import scipy.io as io


class yaml_handle:

    def __init__(self, file, encoding='utf-8'):
        self.file = file
        self.encoding = encoding
        pass

    def init_yaml(self):
        """
        description:
            创建空白.yaml文件
        :param:
            none
        :return:
            yaml file
        """
        file = open(self.file, "w")
        file.close()

    def get_yaml(self):
        """
        description:
            读取.yaml文件的数据
        :param:
            none
        :return:
            List
        """
        with open(self.file, encoding=self.encoding) as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        return data

    def add_yaml(self, name, id, rows, lines, M_data):
        """
        description:
            往指定的.yaml文件的添加数据
            * 若是没有此.yaml文件，则会根据文件名新建一个。省略掉初始化函数。
        :param:
            name: N Frames,id:id,rows:Matrix's rows,lines:Matrix's lines,M_data:N frames's Matrix.
        :return:
        """
        data = {
            name: {
                "id": id,
                "rows": rows,
                "lines": lines,
                "data": {
                    "r00": M_data[0],
                    "r01": M_data[1],
                    "r02": M_data[2],
                    "r03": M_data[3],
                    "r10": M_data[4],
                    "r11": M_data[5],
                    "r12": M_data[6],
                    "r13": M_data[7],
                    "r20": M_data[8],
                    "r21": M_data[9],
                    "r22": M_data[10],
                    "r23": M_data[11]
                }
            }
        }
        with open(self.file, 'a', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def conver_yaml(self, data):
        """
        description:
            将.yaml读取到的列表信息:data，转换成数组
        :param:
            data:从.yaml文件读取到的列表信息
        :return:
            Data1:排列好的二维矩阵信息，shape为:(framesNumbers,12)
        """
        frames_number = len(data)  # 对List:data 进行计数
        # print(data["frame_0"]["data"]["r00"])
        Data1 = np.zeros(frames_number * 12)
        Data1 = Data1.reshape(frames_number, 12)
        for i in range(frames_number):
            Data1[i][0] = data[f"frame_{i}"]["data"]["r00"]
            Data1[i][1] = data[f"frame_{i}"]["data"]["r01"]
            Data1[i][2] = data[f"frame_{i}"]["data"]["r02"]
            Data1[i][3] = data[f"frame_{i}"]["data"]["r03"]
            Data1[i][4] = data[f"frame_{i}"]["data"]["r10"]
            Data1[i][5] = data[f"frame_{i}"]["data"]["r11"]
            Data1[i][6] = data[f"frame_{i}"]["data"]["r12"]
            Data1[i][7] = data[f"frame_{i}"]["data"]["r13"]
            Data1[i][8] = data[f"frame_{i}"]["data"]["r20"]
            Data1[i][9] = data[f"frame_{i}"]["data"]["r21"]
            Data1[i][10] = data[f"frame_{i}"]["data"]["r22"]
            Data1[i][11] = data[f"frame_{i}"]["data"]["r23"]
        # print(Data1.shape)
        # Data1 = Data1.reshape(frames_number, 3, 4)
        return Data1


if __name__ == '__main__':

    """
    # 获取和输入基本信息
    """
    # 获取当前脚本所在文件夹的路经
    curpath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路经
    Yaml_name = input("please Fill in the save name of the .yaml file (path): ")
    # yamlpath = os.path.join(curpath, "YamlFiles/experience_data1.yaml")
    yamlpath = os.path.join(curpath, Yaml_name)
    yaml_op1 = yaml_handle(yamlpath)
    yaml_op1.init_yaml()

    """
    # 读取.mat 文件操作，并转换成.yaml文件
    """
    Mat_name = input("please Fill in the path and name of .mat file: ")
    data = io.loadmat(Mat_name)
    Mat_var = input("please Fill in the Variable name: ")
    Data = np.array(data[Mat_var])
    print(Data.shape)
    rows = Data.shape[0]
    lines = Data.shape[1]
    Data = Data.reshape(int(rows / 4), 12)
    dic = {}
    for i in range(int(rows / 4)):
        dic[f"frame_{i}"] = Data[i].tolist()
        yaml_op1.add_yaml(f"frame_{i}", i, 3, 4, dic[f"frame_{i}"])

    """
    # 读取.yaml文件，将里面的data数据转换成能使用的格式
    """
    # data = yaml_op1.get_yaml()  # 读取yaml文件
    # frames_number = len(data)  # 对List:data 进行计数
    # # print(data["frame_0"]["data"]["r00"])
    # Data1 = np.zeros(frames_number * 12)
    # Data1 = Data1.reshape(frames_number, 12)
    # for i in range(frames_number):
    #     Data1[i][0] = data[f"frame_{i}"]["data"]["r00"]
    #     Data1[i][1] = data[f"frame_{i}"]["data"]["r01"]
    #     Data1[i][2] = data[f"frame_{i}"]["data"]["r02"]
    #     Data1[i][3] = data[f"frame_{i}"]["data"]["r03"]
    #     Data1[i][4] = data[f"frame_{i}"]["data"]["r10"]
    #     Data1[i][5] = data[f"frame_{i}"]["data"]["r11"]
    #     Data1[i][6] = data[f"frame_{i}"]["data"]["r12"]
    #     Data1[i][7] = data[f"frame_{i}"]["data"]["r13"]
    #     Data1[i][8] = data[f"frame_{i}"]["data"]["r20"]
    #     Data1[i][9] = data[f"frame_{i}"]["data"]["r21"]
    #     Data1[i][10] = data[f"frame_{i}"]["data"]["r22"]
    #     Data1[i][11] = data[f"frame_{i}"]["data"]["r23"]
    # print(Data1.shape)
    # Data1 = Data1.reshape(frames_number, 3, 4)
    # print(Data1)
