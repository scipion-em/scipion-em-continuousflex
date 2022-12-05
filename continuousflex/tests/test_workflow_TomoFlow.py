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
from continuousflex.protocols.protocol_subtomograms_synthesize import MODE_RELATION_3CLUSTERS, MODE_RELATION_PARABOLA
from continuousflex.protocols.protocol_pdb_dimred import FlexProtDimredPdb
from continuousflex.protocols.protocol_subtomogram_averaging import FlexProtSubtomogramAveraging, IMPORT_XMIPP_MD, COPY_STA
from continuousflex.protocols.protocol_tomoflow import FlexProtHeteroFlow
from continuousflex.protocols.protocol_tomoflow_dimred import FlexProtDimredHeteroFlow
from continuousflex.protocols.protocol_tomoflow_refine_alignment import FlexProtRefineSubtomoAlign
from xmipp3.protocols import XmippProtCreateMask3D


class TestTomoFlow(TestWorkflow):
    """ TomoFlow pipeline test """

    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def test_all(self):
        """ Run NMA then synthesize sybtomograms"""

        # ------------------------------------------------
        # Import a Pdb -> NMA
        # ------------------------------------------------
        # Import a PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('Import test model')
        self.launchProtocol(protImportPdb)

        # Launch NMA for PDB imported
        protNMA = self.newProtocol(FlexProtNMA,
                                   cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protImportPdb.outputPdb)
        protNMA.setObjLabel('Simulate test movements')
        self.launchProtocol(protNMA)
        # ------------------------------------------------------------------------------------
        # Synthesize subtomograms with 3 clusters relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=6,
                                          modeRelationChoice=MODE_RELATION_3CLUSTERS)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('Test volumes with discrete variability')
        self.launchProtocol(protSynthesize)

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('PCA on groundtruth PDBs')
        self.launchProtocol(protpdbdimred)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   StA_choice=COPY_STA,
                                   import_choice=IMPORT_XMIPP_MD
                                   )
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.xmippMD.set(protSynthesize._getExtraPath('GroundTruth.xmd'))
        protStA.setObjLabel('StA (using groundtruth alignment)')
        self.launchProtocol(protStA)

        # Generate a mask following the shape of the subtomogram average:
        protMask = self.newProtocol(XmippProtCreateMask3D,
                                    source=0, # 0 is SOURCE_VOLUME
                                    inputVolume=protStA.SubtomogramAverage,
                                    threshold = 0.1,
                                    doBig = True,
                                    doMorphological = True,
                                    elementSize = 3,
                                    doSmooth = True
                                    )
        protMask.setObjLabel('Generate ROI mask')
        self.launchProtocol(protMask)

        # Perform missing wedge correction and refinement:
        protRefine = self.newProtocol(FlexProtRefineSubtomoAlign,N_GPU = 1)
        protRefine.NumOfIters.set(1)
        protRefine.inputVolumes.set(protSynthesize.outputVolumes)
        protRefine.STAVolume.set(protStA.SubtomogramAverage)
        protRefine.MetaDataSTA.set(protStA)
        protRefine.applyMask.set(True)
        protRefine.Mask.set(protMask.outputMask)
        protRefine.iterations.set(1)
        protRefine.winsize.set(5)
        protRefine.setObjLabel('MW fill and refine alignment')
        self.launchProtocol(protRefine)

        # Analyze the variability of the optical flows:
        protTomoFlow = self.newProtocol(FlexProtHeteroFlow)
        protTomoFlow.refinementProt.set(protRefine)
        protTomoFlow.setObjLabel('Analyze OF variability')
        self.launchProtocol(protTomoFlow)

        # Generate the conformational space:
        ProtFlowDimred = self.newProtocol(FlexProtDimredHeteroFlow,
                                          reducedDim=3)
        ProtFlowDimred.inputOpFlow.set(protTomoFlow)
        ProtFlowDimred.setObjLabel('TomoFlow Conformational space')
        self.launchProtocol(ProtFlowDimred)

        # ------------------------------------------------------------------------------------
        # Synthesize subtomograms with continuous relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=10,
                                          modeRelationChoice=MODE_RELATION_PARABOLA,
                                          modesAmplitudeRange=100
                                          )
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('Test volumes with continuous variability')
        self.launchProtocol(protSynthesize)

        protpdbdimred = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred.pdbs.set(protSynthesize)
        protpdbdimred.setObjLabel('PCA on groundtruth PDBs')
        self.launchProtocol(protpdbdimred)

        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   StA_choice=COPY_STA,
                                   import_choice=IMPORT_XMIPP_MD
                                   )
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.xmippMD.set(protSynthesize._getExtraPath('GroundTruth.xmd'))
        protStA.setObjLabel('StA (using groundtruth alignment)')
        self.launchProtocol(protStA)

        # Generate a mask following the shape of the subtomogram average:
        protMask = self.newProtocol(XmippProtCreateMask3D,
                                    source=0, # 0 is SOURCE_VOLUME
                                    inputVolume=protStA.SubtomogramAverage,
                                    threshold = 0.1,
                                    doBig = True,
                                    doMorphological = True,
                                    elementSize = 3,
                                    doSmooth = True
                                    )
        protMask.setObjLabel('Generate ROI mask')
        self.launchProtocol(protMask)

        # Perform missing wedge correction and refinement:
        protRefine = self.newProtocol(FlexProtRefineSubtomoAlign,N_GPU = 1)
        protRefine.NumOfIters.set(1)
        protRefine.inputVolumes.set(protSynthesize.outputVolumes)
        protRefine.STAVolume.set(protStA.SubtomogramAverage)
        protRefine.MetaDataSTA.set(protStA)
        protRefine.applyMask.set(True)
        protRefine.Mask.set(protMask.outputMask)
        protRefine.iterations.set(1)
        protRefine.winsize.set(5)
        protRefine.setObjLabel('MW fill and refine alignment')
        self.launchProtocol(protRefine)

        # Analyze the variability of the optical flows:
        protTomoFlow = self.newProtocol(FlexProtHeteroFlow)
        protTomoFlow.refinementProt.set(protRefine)
        protTomoFlow.setObjLabel('Analyze OF variability')
        self.launchProtocol(protTomoFlow)

        # Generate the conformational space:
        ProtFlowDimred = self.newProtocol(FlexProtDimredHeteroFlow,
                                          reducedDim=3)
        ProtFlowDimred.inputOpFlow.set(protTomoFlow)
        ProtFlowDimred.setObjLabel('TomoFlow Conformational space')
        self.launchProtocol(ProtFlowDimred)