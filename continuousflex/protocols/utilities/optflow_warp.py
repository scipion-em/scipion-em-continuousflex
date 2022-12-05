# By Mohamad Harastani

from spider_files3 import save_volume
import sys
import farneback3d
import joblib

def opflow_warp(ref_dump, flow_dump, warped_path_i):
    reference = joblib.load(ref_dump)
    flow_i = joblib.load(flow_dump)
    warped_i = farneback3d.warp_by_flow(reference, flow_i)
    save_volume(warped_i, warped_path_i)


if __name__ == '__main__':
    opflow_warp(sys.argv[1],
                sys.argv[2],
                sys.argv[3])
    sys.exit()