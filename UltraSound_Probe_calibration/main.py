from scipy.optimize import fsolve
import numpy as np
from Others import kabsch, LoadData_mat

p1_raw = np.array([-47.7666, 118.5508, 633.4253,
                   -4.6951, 123.9115, 619.2024,
                   0.3865, 124.4499, 617.3193,
                   -44.6705, 188.0993, 671.5347,
                   -1.4557, 192.6857, 657.7082,
                   2.9381, 193.9377, 655.6947])

p2_raw = np.array([-8.7369, 67.4820, 742.9518,
                   -3.0507, 61.2111, 736.3498,
                   39.9845, 66.0803, 722.5941,
                   -3.7306, 131.6405, 775.2590,
                   1.4531, 132.7506, 774.5312,
                   43.9121, 138.6730, 760.4695])

if __name__ == '__main__':
    # uv3_raw = uv3_raw.reshape(15, 2, 6)
    # p1 = p1_raw.reshape(6, 3).T
    # p2 = p2_raw.reshape(6, 3).T
    # probecamera = probecamera_raw.reshape((45, 4), order='f').T
    # probecamera = probecamera.reshape((15,3,4))
    ultrasound_name1 = '../matFiles/ultrasound/ultrasound_pixel_data.mat'
    var1 = 'uv'
    Data = LoadData_mat.reshapeData_ultrasoundimg(ultrasound_name1,var1)
    print(Data)
    pass
