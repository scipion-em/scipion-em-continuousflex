# **************************************************************************
# * Authors:  Remi Vuillemot             (remi.vuillemot@upmc.fr)
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

        form.addParam('nsteps', IntParam, default="5",
                      label="Number of steps between models",
                      help="")
        form.addParam('loop', IntParam, default=4,
                      label="Number of loop of the movie",
                      help="")

        form.addParam('volumes', BooleanParam, default=True,
                      label="Show maps ?",
                      help="")

        form.addParam('gaussian', BooleanParam, default=True,
                      label="Apply gaussian filter ?",
                      help="", condition="volumes")
        form.addParam('sdev', FloatParam, default=1.0,
                      label="gaussian sigma",
                      help="", condition="gaussian")

        form.addParam('models', BooleanParam, default=True,
                      label="Show models ?",
                      help="")
        form.addParam('fitmap', BooleanParam, default=True,
                      label="Fit models into maps ?",
                      help="", condition="volumes and models")

    def _getVisualizeDict(self):
        return {
                'displayChimera': self._displayChimera,
                }

    def _displayChimera(self, param):

        script_file = self._getTmpPath("cluster_chimerax.cxc")
        makePath(script_file)
        cleanPath(script_file)
        nstep = self.nsteps.get()
        loop = self.loop.get()
        sdev= self.sdev.get()

        pdb_set = self.protocol.inputPDBs.get()
        vol_set = self.protocol.outputVols
        with open(script_file, "w") as f :
            f.write("light full\n")
            f.write("set bgColor white\n")
            f.write("graphics silhouettes true\n")
            start_pdb = 1
            stop_pdb = 0

            if self.models.get():
                for pdb in pdb_set :
                    f.write("open " + os.path.abspath(pdb.getFileName()) + "\n")
                    f.write("color bychain\n")
                    stop_pdb+=1
                # f.write("hide atoms\n")
                # f.write("show cartoons\n")
                f.write("style sphere\n")
                n_pdbs = stop_pdb-start_pdb +1

            start_vol = stop_pdb+1
            stop_vol = stop_pdb
            if self.volumes.get():
                for vol in vol_set :
                    f.write("open " + os.path.abspath(vol.getFileName()) + "\n")
                    stop_vol+=1
                n_vols = stop_vol-start_vol +1

            if self.gaussian.get() and self.volumes.get():
                for i in range(n_vols):
                    f.write("vop gaussian #%i sdev %f \n"%(start_vol+i, sdev))
                start_vol+=n_vols
                stop_vol+=n_vols

            if self.volumes.get():
                f.write("volume voxelSize %f origin %i transparency 0.5 color lightgrey \n"%(vol_set.getSamplingRate(), -vol_set.getXDim()//2))

            if (self.models.get() and self.volumes.get()) and self.fitmap.get():
                for i in range(n_pdbs):
                    f.write("fitmap #%i inMap #%i \n"%(start_pdb+i, start_vol+i))



            if self.models.get():
                    f.write("morph #%i-%i frames %i same true\n"%(start_pdb, stop_pdb, nstep))

            if self.volumes.get():
                nframes = 1+((n_vols-1)*nstep)
                f.write("volume morph #%i-%i playStep %f frames %i ; "%(start_vol, stop_vol, 1/(nframes-0.5), (2*nframes)*loop))
            if self.models.get():
                f.write("coordset #%i loop %i bounce true\n"%(stop_vol+1, loop))
            else:
                f.write("\n")

            f.write("hide #%i-%i \n"%(start_pdb, stop_vol))

            # f.write("hide atoms\n")
            # f.write("show cartoons\n")

        cv = ChimeraView(script_file)
        cv.show()
