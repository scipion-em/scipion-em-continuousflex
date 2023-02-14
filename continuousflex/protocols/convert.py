# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# * Slavica Jonic (slavica.jonic@upmc.fr)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from collections import OrderedDict
from pwem.emlib import (MDL_NMA_MODEFILE, MDL_NMA_COLLECTIVITY, MDL_NMA_SCORE, MDL_ORDER)
from pyworkflow.utils import Environ
from pwem.objects import NormalMode

from xmipp3.convert import rowToObject, objectToRow
from continuousflex.constants import NMA_HOME
import numpy as np
import math

            
MODE_DICT = OrderedDict([ 
       ("_modeFile", MDL_NMA_MODEFILE),
       ("_collectivity", MDL_NMA_COLLECTIVITY),
       ("_score", MDL_NMA_SCORE),
])


def rowToMode(row):
    """ Set properties of a NormalMode object from a Metadata row. """
    mode = NormalMode()
    rowToObject(row, mode, MODE_DICT)
    mode.setObjId(row.getValue(MDL_ORDER))
    return mode


def modeToRow(mode, row):
    """ Write the MetaData row from a given NormalMode object. """
    row.setValue(MDL_ORDER, int(mode.getObjId()))
    objectToRow(mode, row, MODE_DICT)
    
    
def getNMAEnviron():
    """ Create the needed environment for NMA programs. """
    from continuousflex import Plugin
    environ = Plugin.getEnviron()
    environ.update({'PATH': Plugin.getVar(NMA_HOME)}, position=Environ.BEGIN)
    environ.update({'LD_LIBRARY_PATH': Plugin.getCondaLibPath()}, position=Environ.BEGIN)
    return environ


def eulerAngles2matrix(alpha, beta, gamma, shiftx, shifty, shiftz):
    A = np.empty([4,4])
    A.fill(2)
    A[3,3] = 1
    A[3,0:3] = 0
    A[0,3] = float(shiftx)
    A[1,3] = float(shifty)
    A[2,3] = float(shiftz)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    sa = np.sin(np.deg2rad(alpha))
    ca = np.cos(np.deg2rad(alpha))
    sb = np.sin(np.deg2rad(beta))
    cb = np.cos(np.deg2rad(beta))
    sg = np.sin(np.deg2rad(gamma))
    cg = np.cos(np.deg2rad(gamma))
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    A[0,0] = cg * cc - sg * sa
    A[0,1] = cg * cs + sg * ca
    A[0,2] = -cg * sb
    A[1,0] = -sg * cc - cg * sa
    A[1,1] = -sg * cs + cg * ca
    A[1,2] = sg * sb
    A[2,0] = sc
    A[2,1] = ss
    A[2,2] = cb
    return A


def matrix2eulerAngles(A):
    abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
    if (abs_sb > 16*np.exp(-5)):
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
            beta  = 0
            gamma = math.atan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0
            beta  = np.pi
            gamma = math.atan2(A[1, 0], -A[0, 0])
    gamma = np.rad2deg(gamma)
    beta  = np.rad2deg(beta)
    alpha = np.rad2deg(alpha)
    return alpha, beta, gamma, A[0,3], A[1,3], A[2,3]


def l2(Vec1, Vec2):
    Vec1 = np.array(Vec1)
    Vec2 = np.array(Vec2)
    value = np.inner(Vec1-Vec2, Vec1-Vec2)
    return np.sqrt(value)
