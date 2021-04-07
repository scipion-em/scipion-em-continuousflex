# **************************************************************************
# * Authors:     Mohamad Harastani (mohamad.harastani@upmc.fr)
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
from continuousflex.protocols import FlexProtAlignmentNMAVol, FlexProtDimredNMAVol
from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, NMA_CUTOFF_ABS,
                                      FlexProtConvertToPseudoAtoms)
from continuousflex.protocols.pdb.protocol_pseudoatoms_base import NMA_MASK_THRE
from continuousflex.protocols.protocol_nma_dimred_vol import DIMRED_SKLEAN_PCA
import os

from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pwem import Domain
from pyworkflow.tests import setupTestProject, DataSet

from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo, NMA_CUTOFF_ABS)
from continuousflex.protocols.protocol_subtomogrmas_synthesize import MODE_RELATION_LINEAR, MODE_RELATION_3CLUSTERS, \
    MODE_RELATION_MESH, MODE_RELATION_RANDOM
from continuousflex.protocols.protocol_pdb_dimred import FlexProtDimredPdb
from continuousflex.protocols.protocol_subtomograms_classify import FlexProtSubtomoClassify
from continuousflex.protocols.protocol_subtomogram_averaging import FlexProtSubtomogramAveraging
from continuousflex.protocols.protocol_missing_wedge_filling import FlexProtMissingWedgeFilling


class TestHEMNMA3D(TestWorkflow):
    """ Check the images are converted properly to spider format. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma')

    def test_nma3D(self):
        """ Run NMA simple workflow for both Atomic and Pseudoatoms. """
        # ------------------------------------------------
        # Case 1. Import a Pdb -> NMA
        # ------------------------------------------------
        # Import a PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('AK.pdb')
        self.launchProtocol(protImportPdb)
        # Launch NMA for PDB imported
        protNMA = self.newProtocol(FlexProtNMA,
                                   cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protImportPdb.outputPdb)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)
        SNR = 0.1
        N = 3
        M = 6
        # Synthesize subtomograms with 3 clusters relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=N,
                                          modeRelationChoice=MODE_RELATION_3CLUSTERS,
                                          targetSNR=SNR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('subtomograms 3 clusters')
        self.launchProtocol(protSynthesize)
        # Launch HEMNMA-3D
        protAlignment = self.newProtocol(FlexProtAlignmentNMAVol,
                                         modeList='7-9')
        protAlignment.inputModes.set(protNMA.outputModes)
        protAlignment.inputVolumes.set(protSynthesize.outputVolumes)
        protAlignment.setObjLabel('HEMNMA-3D atomic ref')
        self.launchProtocol(protAlignment)
        # Launch Dimred after HEMNMA-3D alignment
        protDimRed = self.newProtocol(FlexProtDimredNMAVol,
                                      dimredMethod=DIMRED_SKLEAN_PCA,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        protDimRed.setObjLabel('HEMNMA-3D dimred')
        self.launchProtocol(protDimRed)
        # ------------------------------------------------
        # Case 2. Import Vol -> Pdb -> NMA
        # ------------------------------------------------
        # Import a Volume
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         filesPath=self.ds.getFile('vol'),
                                         samplingRate=1.0)
        protImportVol.setObjLabel('AK (EM map)')
        self.launchProtocol(protImportVol)
        # Convert the Volume to Pdb
        protConvertVol = self.newProtocol(FlexProtConvertToPseudoAtoms)
        protConvertVol.inputStructure.set(protImportVol.outputVolume)
        protConvertVol.maskMode.set(NMA_MASK_THRE)
        protConvertVol.maskThreshold.set(0.2)
        protConvertVol.pseudoAtomRadius.set(2.5)
        protConvertVol.setObjLabel('AK pseudoatomic structure')
        self.launchProtocol(protConvertVol)
        # Launch NMA with Pseudoatoms
        protNMA2 = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA2.inputStructure.set(protConvertVol.outputPdb)
        self.launchProtocol(protNMA2)

        # Launch HEMNMA-3D
        protAlignment = self.newProtocol(FlexProtAlignmentNMAVol,
                                         modeList='7-9')
        protAlignment.inputModes.set(protNMA2.outputModes)
        protAlignment.inputVolumes.set(protSynthesize.outputVolumes)
        protAlignment.setObjLabel('HEMNMA-3D pseudoatomic ref')
        self.launchProtocol(protAlignment)
        # Launch Dimred after NMA alignment
        protDimRed = self.newProtocol(FlexProtDimredNMAVol,
                                      dimredMethod=DIMRED_SKLEAN_PCA,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        protDimRed.setObjLabel('HEMNMA-3D dimred')
        self.launchProtocol(protDimRed)
