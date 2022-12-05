# **************************************************************************
# *
# * Author:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# * IMPMC, UPMC, Sorbonne University
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

from pwem.protocols import ProtImportPdb
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo,NMA_CUTOFF_ABS)
from continuousflex.protocols.protocol_subtomograms_synthesize import MODE_RELATION_MESH
from continuousflex.protocols.protocol_pdb_dimred import FlexProtDimredPdb
from continuousflex.protocols.protocol_subtomograms_classify import FlexProtSubtomoClassify


class TestSubtomogramSynthesize(TestWorkflow):
    """ subtomogram synthesize test """
    
    @classmethod
    def setUpClass(cls):    
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')
    
    def test_synthesize_all(self):
        """ Run NMA then synthesize sybtomograms"""
        
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

        # Synthesize subtomograms with Mesh relationship
        protSynthesize3 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_MESH)
        protSynthesize3.inputModes.set(protNMA.outputModes)
        protSynthesize3.setObjLabel('synthesized mesh')
        self.launchProtocol(protSynthesize3)

        protpdbdimred3 = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred3.pdbs.set(protSynthesize3)
        protpdbdimred3.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred3)

        protclassifyhierarchical3= self.newProtocol(FlexProtSubtomoClassify,
                                                   numOfClasses=3)
        protclassifyhierarchical3.ProtSynthesize.set(protSynthesize3)
        protclassifyhierarchical3.setObjLabel('hierarchical')
        self.launchProtocol(protclassifyhierarchical3)
        protclassifyKmeans3 = self.newProtocol(FlexProtSubtomoClassify,
                                                     numOfClasses=3,
                                                     classifyTechnique=1,
                                                     reducedDim=3)
        protclassifyKmeans3.ProtSynthesize.set(protSynthesize3)
        protclassifyKmeans3.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans3)

        