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
        data = {
            name: {
                "id": id,
                "rows": rows,
                "lines": lines,
                "data": M_data
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
    yamlpath = os.path.join(curpath, "experience_data.yaml")

    test_yaml = yaml_handle(yamlpath)
    # test_yaml.init_yaml()

    data = io.loadmat('./Tip_calibration/Tracedata.mat')
    Data = np.array(data['Tracedata'])
    Data = Data.reshape(253, 12)

    dic = {}
    for i in range(253):
        dic[f'frame_{i}'] = Data[i].tolist()
        # print(dic[f'frame_{i}'][0])

        # test_yaml.add_yaml(f'frame_{i}', i, 3, 4, dic[f'frame_{i}'])

    # test_yaml.add_yaml('frame', 0, 3, 4, dicJson)
    data = test_yaml.get_yaml()
    # print(data["frame_0"]["data"])
    Data1 = np.zeros(253 * 12)
    Data1 = Data1.reshape(253, 1, 12)
    for i in range(253):
        Data1[i] = data[f"frame_{i}"]["data"]
    print(Data1.shape)
    Data1 = Data1.reshape(253, 3, 4)
    print(Data1)

    pass