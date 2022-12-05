# By Mohamad Harastani

from spider_files3 import open_volume, save_volume

import time
import numpy as np
import sys
import os


def opflow_vols(path_vol0, path_vol1, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, factor1=100,
                factor2=100, path_volx='x_OF_3D.vol', path_voly='y_OF_3D.vol', path_volz='z_OF_3D.vol', gpu_id=0):
    # Convention here is in reverse order
    vol0 = open_volume(path_vol0)
    vol1 = open_volume(path_vol1)

    vol0 = vol0 * factor1
    vol1 = vol1 * factor2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import pycuda.autoinit
    import farneback3d

    optflow = farneback3d.Farneback(
        pyr_scale=pyr_scale,  # Scaling between multi-scale pyramid levels
        levels=levels,  # Number of multi-scale levels
        winsize=winsize,  # Window size for Gaussian filtering of polynomial coefficients
        num_iterations=iterations,  # Iterations on each multi-scale level
        poly_n=poly_n,  # Size of window for weighted least-square estimation of polynomial coefficients
        poly_sigma=poly_sigma,  # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
    )
    t0 = time.time()
    # perform OF:
    flow = optflow.calc_flow(vol0, vol1)
    t_end = time.time()
    print("spent on calculating 3D optical flow", np.floor((t_end - t0) / 60), "minutes and",
          np.round(t_end - t0 - np.floor((t_end - t0) / 60) * 60), "seconds")

    # Extracting the flows in x, y and z dimensions:
    Flowx = flow[0, :, :, :]
    Flowy = flow[1, :, :, :]
    Flowz = flow[2, :, :, :]

    save_volume(Flowx, path_volx)
    save_volume(Flowy, path_voly)
    save_volume(Flowz, path_volz)


if __name__ == '__main__':
    if len(sys.argv) < 9 or len(sys.argv) > 15:
        print('optical flow will not be calculated due to wrong arguments')
    else:
        opflow_vols(sys.argv[1],
                    sys.argv[2],
                    float(sys.argv[3]),
                    int(sys.argv[4]),
                    int(sys.argv[5]),
                    int(sys.argv[6]),
                    int(sys.argv[7]),
                    float(sys.argv[8]),
                    int(sys.argv[9]),
                    int(sys.argv[10]),
                    sys.argv[11],
                    sys.argv[12],
                    sys.argv[13],
                    int(sys.argv[14])
                    )
    sys.exit()