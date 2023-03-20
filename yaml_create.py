import LoadData_mat
import yaml
import os
import numpy as np
import scipy.io as io
import json


class yaml_handle:

    def __init__(self, file, encoding='utf-8'):
        self.file = file
        self.encoding = encoding
        pass

    # def init_yaml(self):
    #     """
    #     description:
    #         初始化创建一个.yaml模板文件
    #     :param:
    #         none
    #     :return:
    #         data
    #     """
    #
    #     init = {
    #         "version": 1.0,
    #         "measure_data": {
    #             "id": 0,
    #             "rows": 3,
    #             "lines": 4,
    #             "data": 0,
    #         },
    #
    #     }
    #     # 初始化操作
    #     with open(self.file, 'w', encoding='utf-8') as f:
    #         yaml.dump(init, f, allow_unicode=True)
    #         f.close()

    def get_yaml(self):
        """
        description:
            读取.yaml文件的数据
        :param:
            none
        :return:
            data
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


if __name__ == '__main__':

    #
    # # print(data_trsm)
    # curpath = os.path.dirname(os.path.realpath(__file__))
    # yamlpath = os.path.join(curpath, "caps.yaml")
    # for i in range(253):
    #     data_trsm = str(json.loads(dicJson)[f"frame_{i}"])
    #     # print(data_trsm,type(data_trsm))
    #
    #     measures_table = {
    #         "measure_data":
    #             {
    #                 'row': 4,
    #                 'cols': 3,
    #                 'frame_id': i,
    #                 'data': {
    #                     data_trsm
    #                 }
    #             }
    #     }
    #     if i == 0:
    #         with open(yamlpath, "w", encoding="utf-8") as f:
    #             yaml.dump(measures_table, f)
    #     else:
    #         with open(yamlpath, "a", encoding="utf-8") as f:
    #             yaml.dump(measures_table, f)

    # 获取当前脚本所在文件夹的路经
    curpath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路经
    yamlpath = os.path.join(curpath, "experience_data1.yaml")

    test_yaml = yaml_handle(yamlpath)
    # test_yaml.init_yaml()

    data = io.loadmat('./Tip_calibration/Tracedata.mat')
    Data = np.array(data['Tracedata'])
    Data = Data.reshape(253, 12)

    # dic = {}
    # for i in range(253):
    #     dic[f'frame_{i}'] = Data[i].tolist()
    # print(dic[f'frame_{i}'][0])
    # print(dic[f'frame_{i}'][0])

    # test_yaml.add_yaml(f'frame_{i}', i, 3, 4, dic[f'frame_{i}'])

    data = test_yaml.get_yaml()
    frames_number = len(data)  # 对List:data 进行计数
    # print(data["frame_0"]["data"]["r00"])
    Data1 = np.zeros(253 * 12)
    Data1 = Data1.reshape(253,12)
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
    print(Data1.shape)
    Data1 = Data1.reshape(253, 3, 4)
    print(Data1)

    pass
