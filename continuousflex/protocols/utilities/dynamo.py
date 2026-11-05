# This function converts a Dynamo table into xmipp (scipion) metadata file
# given a metadata input containing a list of the subtomograms
from pwem.emlib import metadata as md
import numpy as np
from math import sin, cos, radians
from continuousflex.protocols.convert import matrix2eulerAngles
import pandas as pd


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
