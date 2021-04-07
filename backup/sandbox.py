import sys, argparse
import numpy as np, os
import cv2
from PIL import ImageColor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

sys.path.append(".")

from utils.general import check_runs
from feeder.cgan_feeder import Feeder



out        = check_runs('synthetic')
if not os.path.exists(out): os.makedirs(out)

def rotation(data, alpha=0, beta=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

        rx = np.array([[1, 0, 0],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 1, 0],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])

        r = ry.dot(rx)
        data = data.dot(r)

        return data


def normal_skeleton(data):
    #  use as center joint
    center_joint = data[:, 0, :]

    center_jointx = np.mean(center_joint[:, 0])
    center_jointy = np.mean(center_joint[:, 1])
    center_jointz = np.mean(center_joint[:, 2])

    center = np.array([center_jointx, center_jointy, center_jointz])
    data = data - center

    return data

def gaussian_filter(data):
    T, V, C = data.shape

    for v in range(V):
        for c in range(C):
            data[:, v, c] = gaussian_filter1d(data[:, v, c],0.001)

    return data

trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]




parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to generated samples")
parser.add_argument("--index_sample", nargs='+', type=int, default=-1, help="Sample's index")
parser.add_argument("--time", type=int, default=64, help="Re-adjust padding limit from time")  # In case the gan was trained with padding on time
parser.add_argument("--joints", type=int, default=25, help="Re-adjust padding limit from joints")  # In case the gan was trained with padding on joints
opt = parser.parse_args()
print(opt)

data = np.load(opt.path, mmap_mode='r')

print('Data shape', data.shape)

data_numpy = np.array([np.transpose(data[index,:,:opt.time,4:opt.joints+4], (1, 2, 0)) for index in opt.index_sample])
#data_numpy = cv2.normalize(data_numpy, None, alpha=dataset.min, beta=dataset.max, norm_type = cv2.NORM_MINMAX)
data_numpy = np.array([rotation(d, 0,50) for d in data_numpy])
data_numpy = np.array([normal_skeleton(d) for d in data_numpy])
data_numpy = np.array([gaussian_filter(d) for d in data_numpy])

print(data_numpy.shape)
print(data_numpy.max())
print(data_numpy.min())


I, T, V, _ = data_numpy.shape
init_horizon=-45
init_vertical=20
