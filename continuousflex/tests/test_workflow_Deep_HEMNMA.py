# **************************************************************************
# *
# * Authors:     Ilyes Hamitouche (ilyes.hamitouche@upmc.fr)
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
# *
# **************************************************************************

from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtSubSet
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, FlexProtAlignmentNMA,
                                      FlexProtDimredNMA, NMA_CUTOFF_ABS,
                                      FlexProtDeepHEMNMATrain,
                                      FlexProtDeepHEMNMAInfer)
from xmipp3.protocols import XmippProtCropResizeParticles


class TestDeepHEMNMA(TestWorkflow):
    """ Test protocol for deepHEMNMA. """
    @classmethod
    def setUpClass(cls):    
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')
    
    def test_HEMNMA_atomic(self):
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

        protResizeParts= self.newProtocol(XmippProtCropResizeParticles)
        protResizeParts.doResize.set(True)
        protResizeParts.resizeOption.set(2) # this corresponds to factor
        protResizeParts.resizeFactor.set(0.25)
        protResizeParts.inputParticles.set(protImportParts.outputParticles)
        protResizeParts.setObjLabel('Resizing (factor 0.25)')
        self.launchProtocol(protResizeParts)

        protSubset1 = self.newProtocol(ProtSubSet,
                                      objLabel='Training set',
                                      chooseAtRandom=True,
                                      nElements=4)
        # protSubset1.inputFullSet.set(protImportParts.outputParticles)
        protSubset1.inputFullSet.set(protResizeParts.outputParticles)
        self.launchProtocol(protSubset1)


        protSubset2 = self.newProtocol(ProtSubSet,
                                         objLabel='Inference set',
                                         chooseAtRandom=False,
                                         setOperation=1)
        protSubset2.inputFullSet.set(protResizeParts.outputParticles)
        # protSubset2.inputFullSet.set(protImportParts.outputParticles)
        protSubset2.inputSubSet.set(protSubset1.outputParticles)
        self.launchProtocol(protSubset2)

        # Launch NMA alignment
        protAlignment = self.newProtocol(FlexProtAlignmentNMA,
                                         modeList='7-9')
        protAlignment.inputModes.set(protNMA1.outputModes)
        protAlignment.inputParticles.set(protSubset1.outputParticles)
        protAlignment.setObjLabel('HEMNMA atomic ref')
        self.launchProtocol(protAlignment)

        protTrain = self.newProtocol(FlexProtDeepHEMNMATrain)
        protTrain.inputNMA.set(protAlignment)
        self.launchProtocol(protTrain)

        protInfer = self.newProtocol(FlexProtDeepHEMNMAInfer)
        protInfer.trained_model.set(protTrain) #angles and shifts
        protInfer.inputParticles.set(protSubset2.outputParticles)
        self.launchProtocol(protInfer)
