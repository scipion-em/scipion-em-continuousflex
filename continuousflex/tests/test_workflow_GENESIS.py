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

from pwem.protocols import ProtImportPdb, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols.protocol_generate_topology import ProtGenerateTopology
from continuousflex.protocols import FlexProtNMA, NMA_CUTOFF_ABS
from continuousflex.viewers.viewer_genesis import *
from continuousflex.protocols.utilities.pdb_handler import ContinuousFlexPDBHandler


class TestGENESIS(TestWorkflow):
    """ Test Class for GENESIS. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')
        # Import Target EM map
        protImportVol = cls.newProtocol(ProtImportVolumes, importFrom=ProtImportVolumes.IMPORT_FROM_FILES,
                                       filesPath=cls.ds.getFile('1ake_vol'),  samplingRate=2.0)
        protImportVol.setObjLabel('EM map')
        cls.launchProtocol(protImportVol)

        cls.protImportVol = protImportVol
        cls.protPdb4ake = cls.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=cls.ds.getFile('4ake_aa_pdb'))
        cls.protPdb4ake.setObjLabel('Input PDB')
        cls.launchProtocol(cls.protPdb4ake)

    def test1_EmfitVolumeCHARMM(self):

        # Generate topo
        protGenTopo = self.newProtocol(ProtGenerateTopology,
            inputPDB = self.protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CHARMM)
        protGenTopo.setObjLabel('CHARMM topology model')
        self.launchProtocol(protGenTopo)

        # Energy min
        protGenesisMin = self.newProtocol(FlexProtGenesis,
            inputType = INPUT_TOPOLOGY,
            topoProt = protGenTopo,
            simulationType = SIMULATION_MIN,
            n_steps = 100,
            numberOfThreads = NUMBER_OF_CPU,
            numberOfMpi = 1)
        protGenesisMin.setObjLabel('Energy min')
        self.launchProtocol(protGenesisMin)

        # Assert energy descreased
        potential_ene = readLogFile(protGenesisMin.getOutputPrefix()+".log")["POTENTIAL_ENE"]
        assert(potential_ene[0] > potential_ene[-1])

        # Launch NMA for energy min PDB
        protNMA = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protGenesisMin.outputPDB)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)

         # Fit NMMD
        protGenesisFitNMMD = self.newProtocol(FlexProtGenesis,
          inputType=INPUT_RESTART,
          restartProt = protGenesisMin,
          simulationType=SIMULATION_NMMD,
          time_step=0.002,
          n_steps=100,
          nm_number=6,
          nm_mass=1.0,
          inputModes=protNMA.outputModes,
          EMfitChoice=EMFIT_VOLUMES,
          constantK=10000,
          inputVolume=self.protImportVol.outputVolume,
          voxel_size=2.0,
          centerOrigin=True,
          numberOfThreads=NUMBER_OF_CPU,
          numberOfMpi=1,
          )
        protGenesisFitNMMD.setObjLabel('NMMD fitting')
        self.launchProtocol(protGenesisFitNMMD)

        # Assert that the CC is increasing and  the RMSD is decreasing
        cc = readLogFile( protGenesisFitNMMD.getOutputPrefix()+".log")["RESTR_CVS001"]
        inp = ContinuousFlexPDBHandler(protGenesisFitNMMD.getInputPDBprefix() + ".pdb")
        ref = ContinuousFlexPDBHandler(self.ds.getFile('1ake_pdb'))
        out = ContinuousFlexPDBHandler(protGenesisFitNMMD.getOutputPrefix()+".pdb")
        matchingAtoms = inp.matchPDBatoms(reference_pdb=ref)
        rmsd_inp = inp.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)
        rmsd_out = out.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)
        assert(cc[0] < cc[-1])
        assert(rmsd_inp >rmsd_out)

    def test2_EmfitVolumeCAGO(self):
        # Generate topo
        protGenTopo = self.newProtocol(ProtGenerateTopology,
            inputPDB = self.protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CAGO)
        protGenTopo.setObjLabel('C-Alpha Go topology model')
        self.launchProtocol(protGenTopo)

        # energy min
        protGenesisMin = self.newProtocol(FlexProtGenesis,
                                          inputType=INPUT_TOPOLOGY,
                                          topoProt=protGenTopo,
                                          simulationType=SIMULATION_MIN,
                                          numberOfThreads=NUMBER_OF_CPU,
                                          numberOfMpi=1)
        protGenesisMin.setObjLabel('Energy min')
        self.launchProtocol(protGenesisMin)

        # Launch NMA for energy min PDB
        protNMA = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protGenesisMin.outputPDB)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)

        protGenesisFitMD = self.newProtocol(FlexProtGenesis,
                      inputType=INPUT_RESTART,
                      restartProt=protGenesisMin,
                      simulationType=SIMULATION_MD,
                      time_step=0.001,
                      n_steps=1000,
                      temperature=50.0,
                      EMfitChoice=EMFIT_VOLUMES,
                      constantK="500",
                      inputVolume=self.protImportVol.outputVolume,
                      voxel_size=2.0,
                      centerOrigin=True,
                      numberOfThreads=NUMBER_OF_CPU,
                      numberOfMpi=1)
        protGenesisFitMD.setObjLabel('MD fitting')
        self.launchProtocol(protGenesisFitMD)

        # Get GENESIS log file
        log_file = protGenesisFitMD.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Get the RMSD
        inp = ContinuousFlexPDBHandler(protGenesisFitMD.getInputPDBprefix() + ".pdb")
        ref = ContinuousFlexPDBHandler(self.ds.getFile('1ake_pdb'))
        out = ContinuousFlexPDBHandler(protGenesisFitMD.getOutputPrefix()+".pdb")
        matchingAtoms = inp.matchPDBatoms(reference_pdb=ref)
        rmsd_inp = inp.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)
        rmsd_out = out.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)
        assert (cc[0] < cc[-1])
        assert (rmsd_inp > rmsd_out)


        # Need at least 4 cores
        if NUMBER_OF_CPU >= 4:
            protGenesisFitREUS = self.newProtocol(FlexProtGenesis,
                  inputType=INPUT_RESTART,
                  restartProt=protGenesisMin,
                  simulationType=SIMULATION_RENMMD,
                  time_step=0.0005,
                  n_steps=1000,
                  nm_number=6,
                  nm_mass=1.0,
                  inputModes=protNMA.outputModes,
                  exchange_period=100, # 100
                  nreplica = 4,
                  temperature=50.0,
                  constantK="500-1500",
                  EMfitChoice=EMFIT_VOLUMES,
                  inputVolume=self.protImportVol.outputVolume,
                  voxel_size=2.0,
                  centerOrigin=True,
                  numberOfThreads=1,
                  numberOfMpi=NUMBER_OF_CPU,
                  )
            protGenesisFitREUS.setObjLabel('RENMMD fitting')
            self.launchProtocol(protGenesisFitREUS)

            # Get GENESIS log file
            outPref = protGenesisFitREUS.getOutputPrefixAll()
            log_file1 = outPref[0] + ".log"
            log_file2 = outPref[1] + ".log"

            # Get the CC from the log file
            cc1 = readLogFile(log_file1)["RESTR_CVS001"]
            cc2 = readLogFile(log_file2)["RESTR_CVS001"]

            # Get the RMSD
            ref = ContinuousFlexPDBHandler(self.ds.getFile('1ake_pdb'))
            inp = ContinuousFlexPDBHandler(protGenesisFitREUS.getInputPDBprefix() + ".pdb")
            out1 = ContinuousFlexPDBHandler(outPref[0] + ".pdb")
            out2 = ContinuousFlexPDBHandler(outPref[1] + ".pdb")
            matchingAtoms = inp.matchPDBatoms(reference_pdb=ref)
            rmsd_inp = inp.getRMSD(reference_pdb=ref, idx_matching_atoms=matchingAtoms, align=True)
            rmsd_out2 = out2.getRMSD(reference_pdb=ref, idx_matching_atoms=matchingAtoms, align=True)
            rmsd_out1 = out1.getRMSD(reference_pdb=ref, idx_matching_atoms=matchingAtoms, align=True)
            assert (cc1[0] < cc1[-1])
            assert (cc2[0] < cc2[-1])
            assert (rmsd_inp> rmsd_out1)
            assert (rmsd_inp > rmsd_out2)

