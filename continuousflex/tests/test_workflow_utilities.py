# **************************************************************************
# * Authors:     Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * IMPMC, Sorbonne University
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

from pwem.protocols import ProtImportPdb
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import FlexProtSynthesizeSubtomo, FlexProtMissingWedgeRestoration, FlexProtVolumeDenoise


class BM4D_and_MWR(TestWorkflow):
    """ Test protocol for BM4D and MWR. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def test_BM4D(self):
        """ Run NMA simple workflow for both Atomic and Pseudoatoms. """
        # Import PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('AK.pdb')
        self.launchProtocol(protImportPdb)
        SNR = 0.1
        N = 2
        # Synthesize subtomograms
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          confVar=0,
                                          numberOfVolumes=N,
                                          targetSNR=SNR,
                                          volumeSize=32,
                                          samplingRate=4.4,
                                          )
        protSynthesize.refAtomic.set(protImportPdb.outputPdb)
        protSynthesize.setObjLabel('subtomograms')
        self.launchProtocol(protSynthesize)

        # Missing wedge restoration
        protMWC = self.newProtocol(FlexProtMissingWedgeRestoration,
                                   T=5,
                                   )
        protMWC.inputVolumes.set(protSynthesize.outputVolumes)
        protMWC.setObjLabel('missing wedge restoration')
        self.launchProtocol(protMWC)

        # Volume denoising
        protDenoise = self.newProtocol(FlexProtVolumeDenoise)
        protDenoise.inputVolumes.set(protSynthesize.outputVolumes)
        protDenoise.setObjLabel('Bm4D volume denoising')
        self.launchProtocol(protDenoise)
