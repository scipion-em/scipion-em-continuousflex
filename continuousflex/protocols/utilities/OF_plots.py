# By Mohamad Harastani

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly


def plot_quiver_2d(flow, title='optical flow'):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    num_rows = np.shape(flow)[0]
    num_cols = np.shape(flow)[1]
    x = np.arange(0, num_rows, 1)
    y = np.arange(0, num_cols, 1)
    y_pos, x_pos = np.meshgrid(y, x)
    fig, ax = plt.subplots()
    ax.quiver(y_pos, x_pos, v, u)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()


def plot_quiver_3d(flow, downsample=2, title='3D optical flow', normalize=False):
    u = flow[0, :, :, :]
    v = flow[1, :, :, :]
    w = flow[2, :, :, :]
    num_x = np.shape(flow)[1]
    num_y = np.shape(flow)[2]
    num_z = np.shape(flow)[3]

    factors = [(1, downsample), (1, downsample), (1, downsample)]

    for k in range(3):
        u = resample_poly(u, factors[k][0], factors[k][1], axis=k)
        v = resample_poly(v, factors[k][0], factors[k][1], axis=k)
        w = resample_poly(w, factors[k][0], factors[k][1], axis=k)

    x = np.arange(0, int(num_x / downsample), 1)
    y = np.arange(0, int(num_y / downsample), 1)
    z = np.arange(0, int(num_z / downsample), 1)
    x_pos, y_pos, z_pos = np.meshgrid(x, y, z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver3D(x_pos, y_pos, z_pos, u, v, w, length=0.1, normalize=normalize)
    plt.title(title)
    plt.show()
