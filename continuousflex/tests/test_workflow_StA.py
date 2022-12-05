# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
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

from pwem.protocols import ProtImportPdb
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo, NMA_CUTOFF_ABS)
from continuousflex.protocols.protocol_subtomograms_synthesize import MODE_RELATION_LINEAR
from continuousflex.protocols.protocol_subtomograms_classify import FlexProtSubtomoClassify
from continuousflex.protocols.protocol_subtomogram_averaging import FlexProtSubtomogramAveraging
from xmipp3.protocols import XmippProtCreateMask3D
from continuousflex.protocols.protocol_tomoflow_refine_alignment import FlexProtRefineSubtomoAlign


class TestStA(TestWorkflow):
    """ tes for StA protocol """

    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

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
        SNR = 0.1
        N = 30
        # Synthesize subtomograms with 3 clusters relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=N,
                                          modeRelationChoice=MODE_RELATION_LINEAR,
                                          targetSNR=SNR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('subtomograms 3 clusters')
        self.launchProtocol(protSynthesize)

        # Create mask
        protMask = self.newProtocol(XmippProtCreateMask3D,
                                    source=1, # 1 is SOURCE_GEOMETRY
                                    size=64
                                    )
        self.launchProtocol(protMask)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=3,
                                   applyMask=True)
        protStA.Mask.set(protMask.outputMask)
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

        # Apply the alignment on the subtomograms:
        protAlign = self.newProtocol(FlexProtRefineSubtomoAlign,
                                     Alignment_refine = False,
                                     inputVolumes=protSynthesize.outputVolumes,
                                     STAVolume=protStA.SubtomogramAverage,
                                     MetaDataSTA=protStA)
        self.launchProtocol(protAlign)
