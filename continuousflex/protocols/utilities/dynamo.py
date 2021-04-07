# This function converts a Dynamo table into xmipp (scipion) metadata file
# given a metadata input containing a list of the subtomograms
from pwem.emlib import metadata as md
import numpy as np
from math import sin, cos, radians
from xmippLib import Euler_matrix2angles


def dynamo_mat(tdrot, tilt, narot):
    tdrot = radians(tdrot)
    tilt = radians(tilt)
    narot = radians(narot)
    cotd = cos(tdrot)
    sitd = sin(tdrot)
    coti = cos(tilt)
    siti = sin(tilt)
    cona = cos(narot)
    sina = sin(narot)
    m = np.zeros([3,3])
    m[0,0] = cotd * cona - coti * sitd * sina
    m[0,1] = sitd * cona + coti * cotd * sina
    m[0,2] = siti * sina
    m[1,0] = -cotd * sina - coti * sitd * cona
    m[1,1] = -sitd * sina + coti * cotd * cona
    m[1,2] = siti * cona
    m[2,0] = siti * sitd
    m[2,1] = -siti * cotd
    m[2,2] = coti
    return m


def tbl2metadata(table, mdfi, mdfo):
    tbl = np.loadtxt(table)
    # print(tbl.shape)
    x = tbl[:, 3]
    y = tbl[:, 4]
    z = tbl[:, 5]
    tdrot = tbl[:, 6]
    tiltd = tbl[:, 7]
    narot = tbl[:, 8]
    # change the angles from Dynamo convention to xmipp convention
    rot = []
    tilt = []
    psi = []
    for i in range(tbl.shape[0]):
        TransMat = dynamo_mat(tdrot[i],tiltd[i],narot[i])
        # TransMat = np.linalg.inv(TransMat)
        rot_i, tilt_i, psi_i = Euler_matrix2angles(TransMat)
        rot.append(rot_i)
        tilt.append(tilt_i)
        psi.append(psi_i)

    # print(x[0], y[0], z[0], rot[0], tilt[0], psi[0])
    md_out = md.MetaData(mdfi)
    i = 0
    for objId in md_out:
        md_out.setValue(md.MDL_SHIFT_X,    x[i],objId)
        md_out.setValue(md.MDL_SHIFT_Y,    y[i],objId)
        md_out.setValue(md.MDL_SHIFT_Z,    z[i], objId)
        md_out.setValue(md.MDL_ANGLE_ROT,  rot[i], objId)
        md_out.setValue(md.MDL_ANGLE_TILT, tilt[i], objId)
        md_out.setValue(md.MDL_ANGLE_PSI,  psi[i], objId)
        md_out.setValue(md.MDL_ANGLE_Y,    0.0, objId)
        i += 1
    pass
    md_out.write(mdfo)


# a test here:
# mdfni = '/media/mohamad/work/denamo_test/Using_The_Table/volumes_scipion.xmd'
# table = '/media/mohamad/work/denamo_test/Using_The_Table/refined_table_ref_001_ite_0001.tbl'
# mdfno = '/media/mohamad/work/denamo_test/Using_The_Table/result_md.xmd'
# tbl2metadata(table, mdfni,mdfno)
