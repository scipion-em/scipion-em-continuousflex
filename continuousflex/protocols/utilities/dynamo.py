# This function converts a Dynamo table into xmipp (scipion) metadata file
# given a metadata input containing a list of the subtomograms
from pwem.emlib import metadata as md
import numpy as np
from math import sin, cos, radians
from xmippLib import Euler_matrix2angles
import pandas as pd
import math


def dynamo_mat(tdrot, tilt, narot, shiftx, shifty, shiftz):
    tdrot = radians(tdrot)
    tilt = radians(tilt)
    narot = radians(narot)
    cotd = cos(tdrot)
    sitd = sin(tdrot)
    coti = cos(tilt)
    siti = sin(tilt)
    cona = cos(narot)
    sina = sin(narot)
    m = np.zeros([4,4])
    m[0,0] = cotd * cona - sitd * coti * sina
    m[1,0] = - cona * sitd - cotd * coti * sina
    m[2,0] = sina * siti
    m[0,1] = cotd * sina + cona * sitd * coti
    m[1,1] = cotd * cona * coti - sitd * sina
    m[2,1] = -cona * siti
    m[0,2] = sitd * siti
    m[1,2] = cotd * siti
    m[2,2] = coti
    # The 4th column
    m[0,3] = shiftx
    m[1,3] = shifty
    m[2,3] = shiftz
    m[3,3] = 1

    return m


def matrix2eulerAngles(A):
    abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
    if (abs_sb > 16 * np.exp(-5)):
        gamma = math.atan2(A[1, 2], -A[0, 2])
        alpha = math.atan2(A[2, 1], A[2, 0])
        if (abs(np.sin(gamma)) < np.exp(-5)):
            sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
        else:
            if np.sin(gamma) > 0:
                sign_sb = np.sign(A[1, 2])
            else:
                sign_sb = -np.sign(A[1, 2])
        beta = math.atan2(sign_sb * abs_sb, A[2, 2])
    else:
        if (np.sign(A[2, 2]) > 0):
            alpha = 0
            beta = 0
            gamma = math.atan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = math.atan2(A[1, 0], -A[0, 0])
    gamma = np.rad2deg(gamma)
    beta = np.rad2deg(beta)
    alpha = np.rad2deg(alpha)
    return alpha, beta, gamma, A[0, 3], A[1, 3], A[2, 3]


def tbl2metadata(table, mdfi, mdfo):
    tbl = pd.read_csv(table, delimiter=' ', header=None)
    x = tbl[:][3]
    y = tbl[:][4]
    z = tbl[:][5]
    tdrot = tbl[:][6]
    tiltd = tbl[:][7]
    narot = tbl[:][8]
    # change the angles from Dynamo convention to xmipp convention
    rot = []
    tilt = []
    psi = []
    shiftx = []
    shifty = []
    shiftz = []
    for i in range(tbl.shape[0]):
        TransMat = dynamo_mat(tdrot[i], tiltd[i], narot[i], x[i], y[i], z[i])
        rot_i, tilt_i, psi_i, shiftx_i, shifty_i, shiftz_i = matrix2eulerAngles(TransMat)
        rot.append(rot_i)
        tilt.append(tilt_i)
        psi.append(psi_i)
        shiftx.append(shiftx_i)
        shifty.append(shifty_i)
        shiftz.append(shiftz_i)

    md_out = md.MetaData(mdfi)
    i = 0
    for objId in md_out:
        md_out.setValue(md.MDL_SHIFT_X,    shiftx[i],objId)
        md_out.setValue(md.MDL_SHIFT_Y,    shifty[i],objId)
        md_out.setValue(md.MDL_SHIFT_Z,    shiftz[i], objId)
        md_out.setValue(md.MDL_ANGLE_ROT,  rot[i], objId)
        md_out.setValue(md.MDL_ANGLE_TILT, tilt[i], objId)
        md_out.setValue(md.MDL_ANGLE_PSI,  psi[i], objId)
        md_out.setValue(md.MDL_ANGLE_Y,    0.0, objId)
        i += 1
    pass
    md_out.write(mdfo)
