# **************************************************************************
# *
# * Authors:     P. Conesa (pconesa@cnb.csic.es) [1]
# *              J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [2]
# *              Mohamad Harastani (mohamad.harastani@upmc.fr) [3]
# * [1] Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# * [2] SciLifeLab, Stockholm University
# * [3] IMPMC, Sorbonne University
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
from continuousflex.protocols import FlexProtAlignmentNMAVol, FlexProtDimredNMAVol
from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, NMA_CUTOFF_ABS,
                                      FlexProtConvertToPseudoAtoms)
from continuousflex.protocols.pdb.protocol_pseudoatoms_base import NMA_MASK_THRE
from continuousflex.protocols.protocol_nma_dimred_vol import DIMRED_SKLEAN_PCA
import os

class TestHEMNMA3D(TestWorkflow):
    """ Check the images are converted properly to spider format. """

    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma3D')

    def test_nma3D(self):
        """ Run NMA simple workflow for both Atomic and Pseudoatoms. """

        # ------------------------------------------------
        # Case 1. Import a Pdb -> NMA
        # ------------------------------------------------

        # Import a PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        self.launchProtocol(protImportPdb)

        # Launch NMA for PDB imported
        protNMA1 = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA1.inputStructure.set(protImportPdb.outputPdb)
        self.launchProtocol(protNMA1)

        # Import the set of particles
        # (in this order just to be in the middle in the tree)
        protImportParts = self.newProtocol(ProtImportVolumes,
                                           filesPath=self.ds.getFile('particles'),
                                           samplingRate=2.2)
        self.launchProtocol(protImportParts)

        # Launch NMA alignment, but just reading result from a previous metadata
        protAlignment = self.newProtocol(FlexProtAlignmentNMAVol,
                                         modeList='7-9',
                                         copyDeformations=self.ds.getFile('gold_atomic'))
        protAlignment.inputModes.set(protNMA1.outputModes)
        protAlignment.inputVolumes.set(protImportParts.outputVolumes)
        self.launchProtocol(protAlignment)

        # Launch Dimred after NMA alignment
        protDimRed = self.newProtocol(FlexProtDimredNMAVol,
                                      dimredMethod=DIMRED_SKLEAN_PCA,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        self.launchProtocol(protDimRed)

        # ------------------------------------------------
        # Case 2. Import Vol -> Pdb -> NMA
        # ------------------------------------------------

        # Import a Volume
        protImportVol = self.newProtocol(ProtImportVolumes,
                                         filesPath=self.ds.getFile('vol'),
                                         samplingRate=1.0)
        self.launchProtocol(protImportVol)

        # Convert the Volume to Pdb
        protConvertVol = self.newProtocol(FlexProtConvertToPseudoAtoms)
        protConvertVol.inputStructure.set(protImportVol.outputVolume)
        protConvertVol.maskMode.set(NMA_MASK_THRE)
        protConvertVol.maskThreshold.set(0.2)
        protConvertVol.pseudoAtomRadius.set(2.5)
        self.launchProtocol(protConvertVol)

        # Launch NMA with Pseudoatoms
        protNMA2 = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA2.inputStructure.set(protConvertVol.outputPdb)
        self.launchProtocol(protNMA2)

        # Launch NMA alignment, but just reading result from a previous metadata
        protAlignment = self.newProtocol(FlexProtAlignmentNMAVol,
                                         modeList='7-9',
                                         copyDeformations=self.ds.getFile('gold_pseudoatomic'))
        protAlignment.inputModes.set(protNMA2.outputModes)
        protAlignment.inputVolumes.set(protImportParts.outputVolumes)
        self.launchProtocol(protAlignment)
        self.launchProtocol(protAlignment)

        # Launch Dimred after NMA alignment
        protDimRed = self.newProtocol(FlexProtDimredNMAVol,
                                      dimredMethod=DIMRED_SKLEAN_PCA,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        self.launchProtocol(protDimRed)