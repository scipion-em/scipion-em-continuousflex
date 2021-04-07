# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# * IMPMC, UPMC, Sorbonne University
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


class TestStA(TestWorkflow):
    """ Check the full StA protocol """

    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma')

    def test_StA(self):
        """ Run NMA then synthesize sybtomograms"""

        # ------------------------------------------------
        # Import a Pdb -> NMA
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
        # ------------------------------------------------------------------------------------
        SNR = 0.05
        N = 36
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

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred)

        # Post alignment classification (PCA+Kmeans)
        protKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                      numOfClasses=3,
                                      classifyTechnique=1,
                                      reducedDim=3)
        protKmeans.ProtSynthesize.set(protSynthesize)
        protKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling = self.newProtocol(FlexProtMissingWedgeFilling,
                                                   StartingReference=1,
                                                   AlignmentParameters=2)
        protMissingWedgeFilling.STAVolume.set(protKmeans.GlobalAverage)
        protMissingWedgeFilling.MetaDataSTS.set(protSynthesize)
        protMissingWedgeFilling.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling.setObjLabel('MW filling & alignment (ideal)')
        self.launchProtocol(protMissingWedgeFilling)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=4)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('StA')
        self.launchProtocol(protStA)

        # Post alignment classification (PCA+Kmeans)
        protclassifyKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                              SubtomoSource=1,  # this is for StA
                                              numOfClasses=3,
                                              classifyTechnique=1,
                                              reducedDim=3)
        protclassifyKmeans.StA.set(protStA)
        protclassifyKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling2 = self.newProtocol(FlexProtMissingWedgeFilling,
                                                    StartingReference=1,
                                                    AlignmentParameters=1)
        protMissingWedgeFilling2.STAVolume.set(protStA.outputvolume)
        protMissingWedgeFilling2.MetaDataSTA.set(protStA)
        protMissingWedgeFilling2.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling2.setObjLabel('MW filling & alignment (realistic)')
        self.launchProtocol(protMissingWedgeFilling2)
        #######################################################################
        # Synthesize subtomograms with linear relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=N,
                                          modeRelationChoice=MODE_RELATION_LINEAR,
                                          targetSNR=SNR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('subtomograms linear')
        self.launchProtocol(protSynthesize)

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred)

        # Post alignment classification (PCA+Kmeans)
        protKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                      numOfClasses=3,
                                      classifyTechnique=1,
                                      reducedDim=3)
        protKmeans.ProtSynthesize.set(protSynthesize)
        protKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling = self.newProtocol(FlexProtMissingWedgeFilling,
                                                   StartingReference=1,
                                                   AlignmentParameters=2)
        protMissingWedgeFilling.STAVolume.set(protKmeans.GlobalAverage)
        protMissingWedgeFilling.MetaDataSTS.set(protSynthesize)
        protMissingWedgeFilling.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling.setObjLabel('MW filling & alignment (ideal)')
        self.launchProtocol(protMissingWedgeFilling)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=4)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('StA')
        self.launchProtocol(protStA)

        # Post alignment classification (PCA+Kmeans)
        protclassifyKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                              SubtomoSource=1,  # this is for StA
                                              numOfClasses=3,
                                              classifyTechnique=1,
                                              reducedDim=3)
        protclassifyKmeans.StA.set(protStA)
        protclassifyKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling2 = self.newProtocol(FlexProtMissingWedgeFilling,
                                                    StartingReference=1,
                                                    AlignmentParameters=1)
        protMissingWedgeFilling2.STAVolume.set(protStA.outputvolume)
        protMissingWedgeFilling2.MetaDataSTA.set(protStA)
        protMissingWedgeFilling2.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling2.setObjLabel('MW filling & alignment (post StA)')
        self.launchProtocol(protMissingWedgeFilling2)

        ###########################################################################
        # Synthesize subtomograms with Mesh relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          meshRowPoints=M,
                                          modeRelationChoice=MODE_RELATION_MESH,
                                          targetSNR=SNR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('subtomograms mesh')
        self.launchProtocol(protSynthesize)

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred)

        # Post alignment classification (PCA+Kmeans)
        protKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                      numOfClasses=3,
                                      classifyTechnique=1,
                                      reducedDim=3)
        protKmeans.ProtSynthesize.set(protSynthesize)
        protKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling = self.newProtocol(FlexProtMissingWedgeFilling,
                                                   StartingReference=1,
                                                   AlignmentParameters=2)
        protMissingWedgeFilling.STAVolume.set(protKmeans.GlobalAverage)
        protMissingWedgeFilling.MetaDataSTS.set(protSynthesize)
        protMissingWedgeFilling.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling.setObjLabel('MW filling & alignment (ideal)')
        self.launchProtocol(protMissingWedgeFilling)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=4)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('StA')
        self.launchProtocol(protStA)

        # Post alignment classification (PCA+Kmeans)
        protclassifyKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                              SubtomoSource=1,  # this is for StA
                                              numOfClasses=3,
                                              classifyTechnique=1,
                                              reducedDim=3)
        protclassifyKmeans.StA.set(protStA)
        protclassifyKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling2 = self.newProtocol(FlexProtMissingWedgeFilling,
                                                    StartingReference=1,
                                                    AlignmentParameters=1)
        protMissingWedgeFilling2.STAVolume.set(protStA.outputvolume)
        protMissingWedgeFilling2.MetaDataSTA.set(protStA)
        protMissingWedgeFilling2.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling2.setObjLabel('MW filling & alignment (post StA)')
        self.launchProtocol(protMissingWedgeFilling2)
        ###########################################################################
        # Synthesize subtomograms with Mesh relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=N,
                                          modeRelationChoice=MODE_RELATION_RANDOM,
                                          targetSNR=SNR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('subtomograms random')
        self.launchProtocol(protSynthesize)

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred)

        # Post alignment classification (PCA+Kmeans)
        protKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                      numOfClasses=3,
                                      classifyTechnique=1,
                                      reducedDim=3)
        protKmeans.ProtSynthesize.set(protSynthesize)
        protKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling = self.newProtocol(FlexProtMissingWedgeFilling,
                                                   StartingReference=1,
                                                   AlignmentParameters=2)
        protMissingWedgeFilling.STAVolume.set(protKmeans.GlobalAverage)
        protMissingWedgeFilling.MetaDataSTS.set(protSynthesize)
        protMissingWedgeFilling.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling.setObjLabel('MW filling & alignment (ideal)')
        self.launchProtocol(protMissingWedgeFilling)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=4)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('StA')
        self.launchProtocol(protStA)

        # Post alignment classification (PCA+Kmeans)
        protclassifyKmeans = self.newProtocol(FlexProtSubtomoClassify,
                                              SubtomoSource=1,  # this is for StA
                                              numOfClasses=3,
                                              classifyTechnique=1,
                                              reducedDim=3)
        protclassifyKmeans.StA.set(protStA)
        protclassifyKmeans.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans)

        # Missing wedge filling and applying alignment:
        protMissingWedgeFilling2 = self.newProtocol(FlexProtMissingWedgeFilling,
                                                    StartingReference=1,
                                                    AlignmentParameters=1)
        protMissingWedgeFilling2.STAVolume.set(protStA.outputvolume)
        protMissingWedgeFilling2.MetaDataSTA.set(protStA)
        protMissingWedgeFilling2.inputVolumes.set(protSynthesize.outputVolumes)
        protMissingWedgeFilling2.setObjLabel('MW filling & alignment (post StA)')
        self.launchProtocol(protMissingWedgeFilling2)
