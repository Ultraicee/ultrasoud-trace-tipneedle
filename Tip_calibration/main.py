import LoadData_mat
from Tip_calibration import CalNeedleTip

if __name__ == '__main__':
    data_name = input("please input the name of .mat:")
    Variable_name = input("and input the name of Variable:")
    Data = LoadData.reshapeData(data_name, Variable_name)
    CalNeedleTip.calneedletip(Data)
    # LoadData.P3dshow(Data)
