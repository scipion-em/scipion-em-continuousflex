# **************************************************************************
# *
# * Authors:     P. Conesa (pconesa@cnb.csic.es) [1]
# *              J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [2]
# *
# * [1] Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# * [2] SciLifeLab, Stockholm University
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

from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, FlexProtAlignmentNMA,
                                      FlexProtDimredNMA, NMA_CUTOFF_ABS,
                                      FlexProtConvertToPseudoAtoms)
from continuousflex.protocols.pdb.protocol_pseudoatoms_base import NMA_MASK_THRE
from continuousflex.protocols.protocol_nma_base import NMA_CUTOFF_REL
from continuousflex.protocols.protocol_nma_alignment import NMA_ALIGNMENT_PROJ
from xmipp3.protocols import XmippProtCropResizeParticles


class TestHEMNMA_1(TestWorkflow):
    """ Test protocol for HEMNMA (Hybrid Electron Microscopy Normal Mode Analysis). """
    @classmethod
    def setUpClass(cls):    
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')
    
    def test_HEMNMA_atomic(self):
        """ Run NMA simple workflow for both Atomic and Pseudoatoms. """
        #------------------------------------------------
        # Case 1. Import a Pdb -> NMA
        #------------------------------------------------
        # Import a PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('AK.pdb')
        self.launchProtocol(protImportPdb)
        
        # Launch NMA for PDB imported
        protNMA1 = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA1.inputStructure.set(protImportPdb.outputPdb)
        protNMA1.setObjLabel('NMA')
        self.launchProtocol(protNMA1)
        
        # Import the set of particles 
        # (in this order just to be in the middle in the tree)
        protImportParts = self.newProtocol(ProtImportParticles,
                                           filesPath=self.ds.getFile('particles'),
                                           samplingRate=1.0)
        protImportParts.setObjLabel('Particles')
        self.launchProtocol(protImportParts) 

        # Launch NMA alignment, but just reading result from a previous metadata
        protAlignment = self.newProtocol(FlexProtAlignmentNMA,
                                         modeList='7-9',
                                         copyDeformations=self.ds.getFile('gold/images_WS_atoms.xmd'))
        protAlignment.inputModes.set(protNMA1.outputModes)
        protAlignment.inputParticles.set(protImportParts.outputParticles)
        protAlignment.setObjLabel('HEMNMA atomic ref')
        self.launchProtocol(protAlignment)       

        # Launch Dimred after NMA alignment 
        protDimRed = self.newProtocol(FlexProtDimredNMA,
                                      dimredMethod=0,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        protDimRed.setObjLabel('HEMNMA dimred')
        self.launchProtocol(protDimRed)

        #------------------------------------------------
        # Case 2. Import Vol -> Pdb -> NMA
        #------------------------------------------------

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
                                    cutoffMode=NMA_CUTOFF_REL)
        protNMA2.inputStructure.set(protConvertVol.outputPdb)
        protNMA2.setObjLabel('NMA')
        self.launchProtocol(protNMA2)
                                          
        # Launch NMA alignment, but just reading result from a previous metadata
        protAlignment = self.newProtocol(FlexProtAlignmentNMA,
                                         modeList='7-9',
                                         copyDeformations=self.ds.getFile('gold/images_WS_pseudoatoms.xmd'))
        protAlignment.inputModes.set(protNMA2.outputModes)
        protAlignment.inputParticles.set(protImportParts.outputParticles)
        protAlignment.setObjLabel('HEMNMA pseudoatomic ref')
        self.launchProtocol(protAlignment)       
        
        # Launch Dimred after NMA alignment 
        protDimRed = self.newProtocol(FlexProtDimredNMA,
                                      dimredMethod=0,  # PCA
                                      reducedDim=2)
        protDimRed.inputNMA.set(protAlignment)
        protDimRed.setObjLabel('HEMNMA dimred')
        self.launchProtocol(protDimRed)


class TestHEMNMA_2(TestWorkflow):
    """ Test protocol for HEMNMA (Hybrid Electron Microscopy Normal Mode Analysis). """

    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def test_HEMNMA_atomic(self):
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
        protNMA1 = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA1.inputStructure.set(protImportPdb.outputPdb)
        protNMA1.setObjLabel('NMA')
        self.launchProtocol(protNMA1)

        # Import the set of particles
        # (in this order just to be in the middle in the tree)
        protImportParts = self.newProtocol(ProtImportParticles,
                                           filesPath=self.ds.getFile('small_stk'),
                                           samplingRate=1.0)
        self.launchProtocol(protImportParts)

        protResizeParts= self.newProtocol(XmippProtCropResizeParticles)
        protResizeParts.doResize.set(True)
        protResizeParts.resizeOption.set(2) # this corresponds to factor
        protResizeParts.resizeFactor.set(0.5)
        protResizeParts.inputParticles.set(protImportParts.outputParticles)
        protResizeParts.setObjLabel('Resizing (factor 0.5)')
        self.launchProtocol(protResizeParts)

        # Launch NMA alignment, but just reading result from a previous metadata
        protAlignment = self.newProtocol(FlexProtAlignmentNMA,
                                         modeList='7-8',
                                         alignmentMethod=NMA_ALIGNMENT_PROJ)
        protAlignment.inputModes.set(protNMA1.outputModes)
        protAlignment.inputParticles.set(protResizeParts.outputParticles)
        protAlignment.setObjLabel('HEMNMA atomic ref')
        self.launchProtocol(protAlignment)
