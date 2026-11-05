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
from scipy.spatial.transform import Rotation

            
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
    """ 4x4 transform in Xmipp's Euler convention: A = (Rz(alpha) Ry(beta) Rz(gamma))^T """
    A = np.eye(4)
    A[:3, :3] = Rotation.from_euler('ZYZ', [float(alpha), float(beta), float(gamma)],
                                    degrees=True).as_matrix().T
    A[:3, 3] = [float(shiftx), float(shifty), float(shiftz)]
    return A


def matrix2eulerAngles(A):
    A = np.asarray(A, dtype=float)
    alpha, beta, gamma = Rotation.from_matrix(A[:3, :3].T).as_euler('ZYZ', degrees=True)
    return alpha, beta, gamma, A[0, 3], A[1, 3], A[2, 3]

def l2(Vec1, Vec2):
    Vec1 = np.array(Vec1)
    Vec2 = np.array(Vec2)
    value = np.inner(Vec1-Vec2, Vec1-Vec2)
    return np.sqrt(value)
