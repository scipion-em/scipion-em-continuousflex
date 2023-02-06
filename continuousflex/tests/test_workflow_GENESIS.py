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

from pwem.protocols import ProtImportPdb, ProtImportVolumes#, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols.protocol_generate_topology import ProtGenerateTopology
from continuousflex.protocols import FlexProtNMA, NMA_CUTOFF_ABS, FlexProtSynthesizeImages
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
        protImportVol.setObjLabel('Target EM volume (1AKE)')
        cls.launchProtocol(protImportVol)

        cls.protImportVol = protImportVol


    def test1_EmfitVolumeCHARMM(self):
        # Import PDB to fit
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake_aa_pdb'))
        protPdb4ake.setObjLabel('Input PDB (4AKE All-Atom)')
        self.launchProtocol(protPdb4ake)

        # Energy min
        protGenTopo = self.newProtocol(ProtGenerateTopology,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CHARMM,
            inputPRM = self.ds.getFile('charmm_prm'),
            inputRTF = self.ds.getFile('charmm_top'),
            inputPSF=self.ds.getFile('4ake_aa_psf'))
        self.launchProtocol(protGenTopo)


        # Energy min
        protGenesisMin = self.newProtocol(FlexProtGenesis,
            inputType = INPUT_TOPOLOGY,
            topoProt = protGenTopo,

            simulationType = SIMULATION_MIN,
            time_step = 0.002,
            n_steps = 100,
            eneout_period = 10,
            crdout_period = 10,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            numberOfThreads = NUMBER_OF_CPU,
            numberOfMpi = 1,

       )

        protGenesisMin.setObjLabel('Energy Minimization CHARMM')
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # Get GENESIS log file
        output_prefix = protGenesisMin.getOutputPrefix()
        log_file = output_prefix+".log"

        # Get the potential energy from the log file
        potential_ene = readLogFile(log_file)["POTENTIAL_ENE"]

        # Assert that the potential energy is decreasing
        # print("\n\n//////////////////////////////////////////////")
        # print(protGenesisMin.getObjLabel())
        # print("Initial potential energy : %.2f kcal/mol"%potential_ene[0])
        # print("Final potential energy : %.2f kcal/mol"%potential_ene[-1])
        # print("//////////////////////////////////////////////\n\n")

        assert(potential_ene[0] > potential_ene[-1])


        # Launch NMA for energy min PDB
        protNMA = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protGenesisMin.outputPDB)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)

        protGenesisFitNMMD = self.newProtocol(FlexProtGenesis,
          inputType=INPUT_RESTART,
          restartProt = protGenesisMin,

          simulationType=SIMULATION_NMMD,
          time_step=0.002,
          n_steps=100, # 3000
          eneout_period=100,
          crdout_period=100,
          nbupdate_period=10,
          nm_number=6,
          nm_mass=1.0,
          inputModes=protNMA.outputModes,

          implicitSolvent=IMPLICIT_SOLVENT_NONE,
          electrostatics=ELECTROSTATICS_CUTOFF,
          switch_dist=10.0,
          cutoff_dist=12.0,
          pairlist_dist=15.0,

          ensemble=ENSEMBLE_NVT,
          tpcontrol=TPCONTROL_LANGEVIN,
          temperature=300.0,

          boundary=BOUNDARY_NOBC,
          EMfitChoice=EMFIT_VOLUMES,
          constantK=10000,
          emfit_sigma=2.0,
          emfit_tolerance=0.1,
          inputVolume=self.protImportVol.outputVolume,
          voxel_size=2.0,
          centerOrigin=True,

          numberOfThreads=NUMBER_OF_CPU,
          numberOfMpi=1,
          )
        protGenesisFitNMMD.setObjLabel('NMMD Flexible Fitting CHARMM')

        # Launch Fitting
        self.launchProtocol(protGenesisFitNMMD)

        # Get GENESIS log file
        log_file = protGenesisFitNMMD.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Get the RMSD
        inp = ContinuousFlexPDBHandler(protGenesisFitNMMD.getInputPDBprefix() + ".pdb")
        ref = ContinuousFlexPDBHandler(self.ds.getFile('1ake_pdb'))
        out = ContinuousFlexPDBHandler(protGenesisFitNMMD.getOutputPrefix()+".pdb")
        matchingAtoms = inp.matchPDBatoms(reference_pdb=ref)
        rmsd_inp = inp.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)
        rmsd_out = out.getRMSD(reference_pdb=ref,idx_matching_atoms=matchingAtoms,align=True)

        # Assert that the CC is increasing and  the RMSD is decreasing
        # print("\n\n//////////////////////////////////////////////")
        # print(protGenesisFitNMMD.getObjLabel())
        # print("Initial CC : %.2f"%cc[0])
        # print("Final CC : %.2f"%cc[-1])
        # print("Initial rmsd : %.2f Ang"%rmsd_inp)
        # print("Final rmsd : %.2f Ang"%rmsd_out)
        # print("//////////////////////////////////////////////\n\n")

        assert(cc[0] < cc[-1])
        assert(rmsd_inp >rmsd_out)
        # assert(rmsd[-1] < 3.0)

    def test2_EmfitVolumeCAGO(self):
        # Import PDB to fit
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake_ca_pdb'))
        protPdb4ake.setObjLabel('Input PDB (4AKE C-Alpha only)')
        self.launchProtocol(protPdb4ake)


        protGenesisMin = self.newProtocol(FlexProtGenesis,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CAGO,
            inputType = INPUT_NEW_SIM,
            inputTOP = self.ds.getFile('4ake_ca_top'),

            simulationType = SIMULATION_MIN,
            time_step = 0.001,
            n_steps = 100,
            eneout_period = 10,
            crdout_period = 10,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

          numberOfThreads=NUMBER_OF_CPU,
          numberOfMpi=1,
       )
        protGenesisMin.setObjLabel('Energy Minimization CAGO')
        # Launch minimisation
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
                                              time_step=0.0005,
                                              n_steps=1000,
                                              eneout_period=100,
                                              crdout_period=100,
                                              nbupdate_period=10,

                                              implicitSolvent=IMPLICIT_SOLVENT_NONE,
                                              electrostatics=ELECTROSTATICS_CUTOFF,
                                              switch_dist=10.0,
                                              cutoff_dist=12.0,
                                              pairlist_dist=15.0,

                                              ensemble=ENSEMBLE_NVT,
                                              tpcontrol=TPCONTROL_LANGEVIN,
                                              temperature=50.0,

                                              boundary=BOUNDARY_NOBC,
                                              EMfitChoice=EMFIT_VOLUMES,
                                              constantK="500",
                                              emfit_sigma=2.0,
                                              emfit_tolerance=0.1,
                                              inputVolume=self.protImportVol.outputVolume,
                                              voxel_size=2.0,
                                              centerOrigin=True,

                                              numberOfThreads=NUMBER_OF_CPU,
                                              numberOfMpi=1,
                                              )
        protGenesisFitMD.setObjLabel('MD Flexible Fitting CAGO')

        # Launch Fitting
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

        # Assert that the CC is increasing and  the RMSD is decreasing
        # print("\n\n//////////////////////////////////////////////")
        # print(protGenesisFitMD.getObjLabel())
        # print("Initial CC : %.2f"%cc[0])
        # print("Final CC : %.2f"%cc[-1])
        # print("Initial rmsd : %.2f Ang"%rmsd_inp)
        # print("Final rmsd : %.2f Ang"%rmsd_out)
        # print("//////////////////////////////////////////////\n\n")
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
                                                  eneout_period=100,
                                                  crdout_period=100,
                                                  nbupdate_period=10,
                                                  nm_number=6,
                                                  nm_mass=1.0,
                                                  inputModes=protNMA.outputModes,
                                                  exchange_period=100, # 100
                                                  nreplica = 4,

                                              implicitSolvent=IMPLICIT_SOLVENT_NONE,
                                              electrostatics=ELECTROSTATICS_CUTOFF,
                                              switch_dist=10.0,
                                              cutoff_dist=12.0,
                                              pairlist_dist=15.0,

                                              ensemble=ENSEMBLE_NVT,
                                              tpcontrol=TPCONTROL_LANGEVIN,
                                              temperature=50.0,

                                              boundary=BOUNDARY_NOBC,
                                              EMfitChoice=EMFIT_VOLUMES,
                                              constantK="500-1500",
                                              emfit_sigma=2.0,
                                              emfit_tolerance=0.1,
                                              inputVolume=self.protImportVol.outputVolume,
                                              voxel_size=2.0,
                                              centerOrigin=True,

                                                  numberOfThreads=1,
                                                  numberOfMpi=NUMBER_OF_CPU,
                                              )
            protGenesisFitREUS.setObjLabel('NMMD + REUS Flexible Fitting CAGO')

            # Launch Fitting
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

            # Assert that the CCs are increasing
            # print("\n\n//////////////////////////////////////////////")
            # print(protGenesisFitREUS.getObjLabel())
            # print("Initial CC : [%.2f , %.2f]" % (cc1[0],cc2[0]))
            # print("Final CC :[%.2f , %.2f]" % (cc1[-1],cc2[-1]))
            # print("Initial rmsd : [%.2f , %.2f] Ang" % (rmsd_inp,rmsd_inp))
            # print("Final rmsd : [%.2f , %.2f] Ang" % (rmsd_out1,rmsd_out2))
            # print("//////////////////////////////////////////////\n\n")

            assert (cc1[0] < cc1[-1])
            assert (cc2[0] < cc2[-1])
            assert (rmsd_inp> rmsd_out1)
            assert (rmsd_inp > rmsd_out2)

