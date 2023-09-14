# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Remi Vuillemot             (remi.vuillemot@upmc.fr)
# * IMPMC, UPMC Sorbonne University
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
# **************************************************************************

import numpy as np
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam, PointerParam, IntParam, BooleanParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pwem.viewers import ChimeraView
from continuousflex.protocols.protocol_batch_pdb_cluster import FlexBatchProtClusterSet
from pyworkflow.utils.path import cleanPath, makePath, makeTmpPath
import os

class FlexProtBatchPdbCluster(ProtocolViewer):
    """ Visualization of density and PDB clusters
    """
    _label = 'viewer batch pdb cluster'
    _targets = [FlexBatchProtClusterSet]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        form.addParam('displayChimera', LabelParam,
                      label="Display clusters in ChimeraX",
                      help="")

    def _getVisualizeDict(self):
        return {
                'displayChimera': self._displayChimera,
                }

    def _displayChimera(self, param):

        script_file = self._getTmpPath("cluster_chimerax.cxc")
        makePath(script_file)
        cleanPath(script_file)

        pdb_set = self.protocol.inputPDBs.get()
        vol_set = self.protocol.outputVols
        with open(script_file, "w") as f :
            f.write("light full\n")
            f.write("set bgColor white\n")
            models_pdb = 0

            for pdb in pdb_set :
                f.write("open " + os.path.abspath(pdb.getFileName()) + "\n")
                f.write("color bychain\n")
                models_pdb+=1
            # f.write("hide atoms\n")
            # f.write("show cartoons\n")
            f.write("morph #1-%s frames 4"%models_pdb)

            models_vol =models_pdb
            for vol in vol_set :
                f.write("open " + os.path.abspath(vol.getFileName()) + "\n")
            f.write("volume voxelSize %f origin %i \n"%(vol_set.getSamplingRate(), -vol_set.getXDim()//2))
            n_vols = models_vol-models_pdb
            # f.write("hide atoms\n")
            # f.write("show cartoons\n")

        cv = ChimeraView(script_file)
        cv.show()
