# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
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
import os
from os.path import basename
import numpy as np
from pwem.emlib import MetaData, MDL_ORDER
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam, IntParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceBaseExt, replaceExt, exists

from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtNMOF
import xmipp3
import pwem.emlib.metadata as md
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
from joblib import load

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1


class FlexProtNMOFViewer(ProtocolViewer):
    """ Visualization of nmof fitting progress
    """
    _label = 'viewer nmof'
    _targets = [FlexProtNMOF]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayCC', IntParam, default=1,
                      label='Display the cross-correlation per iteration for volume number',
                      help='If you fit only one volume, keep it one'
                           ', Otherwise, give the number of the volume to see the progress'
                           )
        form.addParam('displayRMSD', IntParam, default=1,
                      label="Display the RMSD per iteration for volume number",
                      help='If you fit only one volume, keep it one'
                           ', Otherwise, give the number of the volume to see the progress'
                      )
        form.addParam('displayCC_all', LabelParam,
                      label='Display cross-correlation overlay plot for all volumes',
                           )
        form.addParam('displayRMSD_all', LabelParam,
                      label='Display RMSD overlay plot for all volumes',
                           )
        form.addParam('displayFit', IntParam, default=1,
                      label='Display the Fitting result of volume number',
                      help='If you fit only one volume, keep it one'
                           ', Otherwise, give the number of the volume to see the result'
                           )
        # TODO: display the PDBs corresponding to the fitting iterations of a given volume
        # TODO: use the fitting of all (last iteration of each volume for example) to display a trajectory

    def _getVisualizeDict(self):
        return {'displayCC': self._viewCC,
                'displayRMSD': self._viewRMSD,
                'displayCC_all': self._viewCC_All,
                'displayRMSD_all': self._viewRMSD_All,
                'displayFit': self._viewFit}

    def _viewCC(self, Param):
        cc_mat = np.loadtxt(self.protocol._getExtraPath('cc.txt'))
        if cc_mat.ndim == 1:
            pass
        else:
            cc_mat = cc_mat[self.displayCC.get()-1,:]
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=np.size(cc_mat)))
        plt.plot(cc_mat)
        plt.xticks()
        plt.title('Cross correlation fitting progress')
        plt.xlabel('iteration')
        plt.ylabel('cross correlation')
        plt.show()


    def _viewRMSD(self, Param):
        rmsd_mat = np.loadtxt(self.protocol._getExtraPath('rmsd.txt'))
        if rmsd_mat.ndim == 1:
            pass
        else:
            rmsd_mat = rmsd_mat[self.displayCC.get()-1,:]
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=np.size(rmsd_mat)))
        plt.plot(rmsd_mat)
        plt.title('RMSD fitting progress')
        plt.xlabel('iteration')
        plt.ylabel('RMSD')
        plt.show()

    def _viewCC_All(self, Param):
        cc_mat = np.loadtxt(self.protocol._getExtraPath('cc.txt'))
        ax = plt.figure().gca()
        count = 1
        for line in cc_mat:
            plt.plot(line, label='volume ' + str(count))
            count += 1
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=np.size(line)))
        plt.legend()
        plt.title('Cross correlation fitting progress')
        plt.xlabel('iteration')
        plt.ylabel('cross correlation')
        plt.show()
        pass


    def _viewRMSD_All(self, Param):
        rmsd_mat = np.loadtxt(self.protocol._getExtraPath('rmsd.txt'))
        ax = plt.figure().gca()
        count = 1
        for line in rmsd_mat:
            plt.plot(line, label='volume ' + str(count))
            count += 1
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=np.size(line)))
        plt.legend()
        plt.title('RMSD fitting progress')
        plt.xlabel('iteration')
        plt.ylabel('RMSD')
        plt.show()

    def _viewFit(self, Param):
        # Get the volume name
        volumes_md = []
        if(exists(self.protocol._getExtraPath('aligned.xmd'))):
            volumes_md = md.MetaData(self.protocol._getExtraPath('aligned.xmd'))
        else:
            volumes_md = md.MetaData(self.protocol._getExtraPath('volumes.xmd'))
        volume_fn = volumes_md.getValue(md.MDL_IMAGE, self.displayFit.get())
        # Get the initial PDB
        atomsFn = []
        if(self.protocol.NMA.get()):
            atomsFn = self.protocol.getInputPdb().getFileName()
        else:
            atomsFn = self.protocol.pdb.get().getFileName()
        # Get the target PDB if exists:
        target_PDB = []
        if(self.protocol.do_rmsd.get()):
            pdbs_list = [f for f in glob.glob(self.protocol.targetPDBs.get())]
            pdbs_list.sort()
            target_PDB = pdbs_list[self.displayFit.get()-1]
        # Get the final PDB
        PDBs_md = md.MetaData(self.protocol._getExtraPath('PDBs.xmd'))
        PDB_fn = PDBs_md.getValue(md.MDL_IMAGE, self.displayFit.get())
        os.system('chimera %(volume_fn)s %(atomsFn)s %(target_PDB)s %(PDB_fn)s' % locals())


