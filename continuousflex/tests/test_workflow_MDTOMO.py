# **************************************************************************
# * Authors:     Rémi Vuillemot (remi.vuillemot@upmc.fr)
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

from continuousflex.protocols.protocol_mdtomo import FlexProtMDTOMO
from continuousflex.protocols import FlexProtNMA, NMA_CUTOFF_ABS, FlexProtSynthesizeSubtomo, \
    FlexProtDimredPdb, FlexProtAlignPdb,FlexProtGenesis
from continuousflex.protocols.utilities.genesis_utilities import *

from continuousflex.protocols.protocol_align_pdbs import PDB_SOURCE_OBJECT
from continuousflex.protocols.protocol_pdb_dimred import PDB_SOURCE_ALIGNED, REDUCE_METHOD_PCA, REDUCE_METHOD_UMAP

class TestMDTOMO(TestWorkflow):
    """ Test Class for MDTOMO. """

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.        ds = DataSet.getDataSet('nma_V2.0')

    def test_MDTOMO(self):
        # ------------------------- Import PDB prot --------------------------------
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                      pdbFile=self.ds.getFile('4ake_ca_pdb'))
        protPdb4ake.setObjLabel('Input PDB (4AKE C-Alpha only)')
        self.launchProtocol(protPdb4ake)
        # ------------------------- Genesis Min prot --------------------------------

        protGenesisMin = self.newProtocol(FlexProtGenesis,
                                          inputPDB=protPdb4ake.outputPdb,
                                          forcefield=FORCEFIELD_CAGO,
                                          inputType=INPUT_NEW_SIM,
                                          inputTOP=self.ds.getFile('4ake_ca_top'),

                                          simulationType=SIMULATION_MIN,
                                          time_step=0.001,
                                          n_steps=100,
                                          eneout_period=10,
                                          crdout_period=10,
                                          nbupdate_period=10,

                                          implicitSolvent=IMPLICIT_SOLVENT_NONE,
                                          electrostatics=ELECTROSTATICS_CUTOFF,
                                          switch_dist=10.0,
                                          cutoff_dist=12.0,
                                          pairlist_dist=15.0,

                                          numberOfThreads=NUMBER_OF_CPU,
                                          numberOfMpi=1,
                                          )
        protGenesisMin.setObjLabel('Energy Minimization CAGO')
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # ------------------------- NMA prot --------------------------------
        # Launch NMA for energy min PDB
        protNMA = self.newProtocol(FlexProtNMA,
                                   cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protGenesisMin.outputPDB)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)

        # ------------------------- synth volumes --------------------------------
        target_subtomo = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         inputModes=protNMA.outputModes,
                                         numberOfVolumes=10,
                                         samplingRate=2.0,
                                         modesAmplitudeRange=50,
                                         seedOption=False,
                                         noiseCTFChoice=1,
                                         volumeSize=64,
                                          rotationShiftChoice=1)
        target_subtomo.setObjLabel('Subtomograms')
        self.launchProtocol(target_subtomo)

        # ------------------------- MDTOMO --------------------------------
        protMDTOMO = self.newProtocol(FlexProtMDTOMO,

                                                 inputType=INPUT_RESTART,
                                                 restartProt=protGenesisMin,
                                                 simulationType=SIMULATION_NMMD,
                                                 time_step=0.001,
                                                 n_steps=5000,
                                                 nm_number=6,
                                                 nm_mass=1.0,
                                                 inputModes=protNMA.outputModes,
                                                 temperature=50.0,
                                                 EMfitChoice=EMFIT_VOLUMES,
                                                 constantK="500",
                                                 inputVolume=target_subtomo.outputVolumes,
                                                 voxel_size=2.0,
                                                 centerOrigin=True,
                                                 numberOfThreads=1,
                                                 numberOfMpi=NUMBER_OF_CPU,
                                                 )
        protMDTOMO.setObjLabel('MDTOMO')

        # Launch Fitting
        self.launchProtocol(protMDTOMO)

        # ------------------------- align pdbs --------------------------------
        alignPDBs = self.newProtocol(FlexProtAlignPdb,
                                     pdbSource = PDB_SOURCE_OBJECT,
                                     setOfPDBs = protMDTOMO.outputPDBs,
                                     alignRefPDB=protGenesisMin.outputPDB,
                                     createOutput=False)
        alignPDBs.setObjLabel('Align output PDBs')
        self.launchProtocol(alignPDBs)

        # ------------------------- PCA --------------------------------
        protPca = self.newProtocol(FlexProtDimredPdb,
                                   pdbSource=PDB_SOURCE_ALIGNED,
                                   alignPdbProt=alignPDBs,
                                   method=REDUCE_METHOD_PCA,
                                   reducedDim=3
                                   )
        protPca.setObjLabel('PCA')
        self.launchProtocol(protPca)
        # ------------------------- UMAP --------------------------------
        protUmap = self.newProtocol(FlexProtDimredPdb,
                                   pdbSource=PDB_SOURCE_ALIGNED,
                                   alignPdbProt=alignPDBs,
                                   method=REDUCE_METHOD_UMAP,
                                    reducedDim=3)
        protUmap.setObjLabel('UMAP')
        self.launchProtocol(protUmap)