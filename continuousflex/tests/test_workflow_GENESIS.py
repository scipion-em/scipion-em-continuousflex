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

from continuousflex.protocols.protocol_genesis import *
from continuousflex.viewers.viewer_genesis import *
import os
import multiprocessing

NUMBER_OF_CPU = multiprocessing.cpu_count()//2

class testGENESIS(TestWorkflow):
    """ Test Class for GENESIS. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def testEmfitVolumeCHARMM(self):
        # Import PDB to fit
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake_pdb'))
        protPdb4ake.setObjLabel('Input PDB (4AKE)')
        self.launchProtocol(protPdb4ake)

        # Import Target EM map
        protImportVol = self.newProtocol(ProtImportVolumes, importFrom=ProtImportVolumes.IMPORT_FROM_FILES,
                                       filesPath=self.ds.getFile('1ake_vol'),  samplingRate=2.0)
        protImportVol.setObjLabel('Target EM volume (1AKE)')
        self.launchProtocol(protImportVol)


        protGenesisMin = self.newProtocol(ProtGenesis,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CHARMM,
            generateTop = True,
            inputPRM = self.ds.getFile('charmm_prm'),
            inputRTF = self.ds.getFile('charmm_top'),

            simulationType = SIMULATION_MIN,
            time_step = 0.002,
            n_steps = 100,
            eneout_period = 10,
            crdout_period = 10,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_GBSA,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            numberOfThreads = NUMBER_OF_CPU,

       )

        protGenesisMin.setObjLabel('[GENESIS]\n Energy Minimization CHARMM Implicit solvent')
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # Get GENESIS log file
        output_prefix = protGenesisMin.getOutputPrefix()
        log_file = output_prefix+".log"

        # Get the potential energy from the log file
        potential_ene = readLogFile(log_file)["POTENTIAL_ENE"]

        # Assert that the potential energy is decreasing
        print("\n\n//////////////////////////////////////////////")
        print(protGenesisMin.getObjLabel())
        print("Initial potential energy : %.2f kcal/mol"%potential_ene[0])
        print("Final potential energy : %.2f kcal/mol"%potential_ene[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(potential_ene[0] > potential_ene[-1])

        protGenesisFit = self.newProtocol(ProtGenesis,

            inputPDB = protGenesisMin.outputPDBs,
            forcefield = FORCEFIELD_CHARMM,
            generateTop = False,
            inputPRM = self.ds.getFile('charmm_prm'),
            inputRTF = self.ds.getFile('charmm_top'),
            inputPSF = protGenesisMin.getInputPDBprefix()+".psf",
            restartchoice = True,
            inputRST = protGenesisMin.getOutputPrefix()+".rst",

            simulationType = SIMULATION_MD,
            integrator = INTEGRATOR_VVERLET,
            time_step = 0.002,
            n_steps = 5000, # 5000
            eneout_period = 100,
            crdout_period = 100,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_GBSA,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            ensemble = ENSEMBLE_NVT,
            tpcontrol = TPCONTROL_LANGEVIN,
            temperature = 300.0,

            boundary = BOUNDARY_NOBC,
            EMfitChoice = EMFIT_VOLUMES,
            constantK = 10000,
            emfit_sigma = 2.0,
            emfit_tolerance = 0.1,
            inputVolume = protImportVol.outputVolume,
            voxel_size = 2.0,
            centerOrigin = True,

            numberOfThreads = NUMBER_OF_CPU,
        )
        protGenesisFit.setObjLabel('[GENESIS]\n MD cryo-EM fitting with CHARMM implicit solvent')

        # Launch Fitting
        self.launchProtocol(protGenesisFit)

        # Get GENESIS log file
        log_file = protGenesisFit.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]



        # Get the RMSD from the dcd file
        matchingAtoms = matchPDBatoms([PDBMol(protGenesisFit.getInputPDBprefix() + ".pdb")
                                          , PDBMol(self.ds.getFile('1ake_pdb'))])
        rmsd = rmsdFromDCD(outputPrefix = protGenesisFit.getOutputPrefix(),
                           inputPDB = protGenesisFit.getInputPDBprefix()+".pdb",
                           targetPDB=self.ds.getFile('1ake_pdb'),
                           idx=matchingAtoms,
                           align=False)

        # Assert that the CC is increasing and  the RMSD is decreasing
        print("\n\n//////////////////////////////////////////////")
        print(protGenesisFit.getObjLabel())
        print("Initial CC : %.2f"%cc[0])
        print("Final CC : %.2f"%cc[-1])
        print("Initial rmsd : %.2f Ang"%rmsd[0])
        print("Final rmsd : %.2f Ang"%rmsd[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(cc[0] < cc[-1])
        assert(rmsd[0] > rmsd[-1])
        assert(rmsd[-1] < 3.0)

        protGenesisFitNMMD = self.newProtocol(ProtGenesis,

          inputPDB=protGenesisMin.outputPDBs,
          forcefield=FORCEFIELD_CHARMM,
          generateTop=False,
          inputPRM=self.ds.getFile('charmm_prm'),
          inputRTF=self.ds.getFile('charmm_top'),
          inputPSF=protGenesisMin.getInputPDBprefix() + ".psf",
          restartchoice=True,
          inputRST=protGenesisMin.getOutputPrefix() + ".rst",

          simulationType=SIMULATION_MD,
          integrator=INTEGRATOR_NMMD,
          time_step=0.002,
          n_steps=3000, # 3000
          eneout_period=100,
          crdout_period=100,
          nbupdate_period=10,
          nm_number=6,
          nm_mass=1.0,

          implicitSolvent=IMPLICIT_SOLVENT_GBSA,
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
          inputVolume=protImportVol.outputVolume,
          voxel_size=2.0,
          centerOrigin=True,

          numberOfThreads=NUMBER_OF_CPU,
          )
        protGenesisFitNMMD.setObjLabel('[GENESIS]\n NMMD cryo-EM fitting with CHARMM implicit solvent')

        # Launch Fitting
        self.launchProtocol(protGenesisFitNMMD)

        # Get GENESIS log file
        log_file = protGenesisFitNMMD.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Get the RMSD from the dcd file
        matchingAtoms = matchPDBatoms([PDBMol(protGenesisFit.getInputPDBprefix() + ".pdb")
                                          , PDBMol(self.ds.getFile('1ake_pdb'))])
        rmsd = rmsdFromDCD(outputPrefix = protGenesisFitNMMD.getOutputPrefix(),
                           inputPDB = protGenesisFitNMMD.getInputPDBprefix()+".pdb",
                           targetPDB=self.ds.getFile('1ake_pdb'),
                           idx=matchingAtoms,
                           align=False)

        # Assert that the CC is increasing and  the RMSD is decreasing
        print("\n\n//////////////////////////////////////////////")
        print(protGenesisFitNMMD.getObjLabel())
        print("Initial CC : %.2f"%cc[0])
        print("Final CC : %.2f"%cc[-1])
        print("Initial rmsd : %.2f Ang"%rmsd[0])
        print("Final rmsd : %.2f Ang"%rmsd[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(cc[0] < cc[-1])
        assert(rmsd[0] > rmsd[-1])
        assert(rmsd[-1] < 3.0)


        # Need at least 2 cores
        if NUMBER_OF_CPU >= 2:
            protGenesisFitREUS = self.newProtocol(ProtGenesis,

                                              inputPDB=protGenesisMin.outputPDBs,
                                              forcefield=FORCEFIELD_CHARMM,
                                              generateTop=False,
                                              inputPRM=self.ds.getFile('charmm_prm'),
                                              inputRTF=self.ds.getFile('charmm_top'),
                                              inputPSF=protGenesisMin.getInputPDBprefix() + ".psf",
                                              restartchoice=True,
                                              inputRST=protGenesisMin.getOutputPrefix() + ".rst",

                                              simulationType=SIMULATION_REMD,
                                              integrator=INTEGRATOR_VVERLET,
                                              time_step=0.002,
                                              n_steps=5000, # 5000
                                              eneout_period=100, # 100
                                              crdout_period=100, # 100
                                              nbupdate_period=10,
                                              exchange_period=100, # 100
                                              nreplica = 2,

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
                                              constantK="9000 11000",
                                              emfit_sigma=2.0,
                                              emfit_tolerance=0.1,
                                              inputVolume=protImportVol.outputVolume,
                                              voxel_size=2.0,
                                              centerOrigin=True,

                                              numberOfThreads=NUMBER_OF_CPU//2,
                                              numberOfMpi=2,
                                              )
            protGenesisFitREUS.setObjLabel('[GENESIS]\n REUS (2 replicas) cryo-EM fitting with CHARMM no solvent')

            # Launch Fitting
            self.launchProtocol(protGenesisFitREUS)

            # Get GENESIS log file
            outPref = protGenesisFitREUS.getOutputPrefixAll()
            log_file1 = outPref[0] + ".log"
            log_file2 = outPref[1] + ".log"

            # Get the CC from the log file
            cc1 = readLogFile(log_file1)["RESTR_CVS001"]
            cc2 = readLogFile(log_file2)["RESTR_CVS001"]


            # Get the RMSD from the dcd file
            matchingAtoms = matchPDBatoms([PDBMol(protGenesisFitREUS.getInputPDBprefix() + ".pdb")
                                                  ,PDBMol(self.ds.getFile('1ake_pdb'))])
            rmsd1 = rmsdFromDCD(outputPrefix=outPref[0],
                               inputPDB=protGenesisFitREUS.getInputPDBprefix() + ".pdb",
                               targetPDB=self.ds.getFile('1ake_pdb'),
                               align=False, idx=matchingAtoms)
            rmsd2 = rmsdFromDCD(outputPrefix=outPref[0],
                                inputPDB=protGenesisFitREUS.getInputPDBprefix() + ".pdb",
                                targetPDB=self.ds.getFile('1ake_pdb'),
                                align=False, idx=matchingAtoms)

            # Assert that the CCs are increasing
            print("\n\n//////////////////////////////////////////////")
            print(protGenesisFitREUS.getObjLabel())
            print("Initial CC : [%.2f , %.2f]" % (cc1[0],cc2[0]))
            print("Final CC :[%.2f , %.2f]" % (cc1[-1],cc2[-1]))
            print("Initial rmsd : [%.2f , %.2f] Ang" % (rmsd1[0],rmsd2[0]))
            print("Final rmsd : [%.2f , %.2f] Ang" % (rmsd1[-1],rmsd2[-1]))
            print("//////////////////////////////////////////////\n\n")

            assert (cc1[0] < cc1[-1])
            assert (cc2[0] < cc2[-1])
            assert (rmsd1[0] > rmsd1[-1])
            assert (rmsd1[-1] < 3.0)
            assert (rmsd2[0] > rmsd2[-1])
            assert (rmsd2[-1] < 3.0)

    def testEmfitVolumeCAGO(self):
        # Import PDB to fit
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake_ca_pdb'))
        protPdb4ake.setObjLabel('Input PDB (4AKE C-Alpha only)')
        self.launchProtocol(protPdb4ake)

        # Import Target EM map
        protImportVol = self.newProtocol(ProtImportVolumes, importFrom=ProtImportVolumes.IMPORT_FROM_FILES,
                                       filesPath=self.ds.getFile('1ake_vol'),  samplingRate=2.0)
        protImportVol.setObjLabel('Target EM volume (1AKE)')
        self.launchProtocol(protImportVol)

        protGenesisMin = self.newProtocol(ProtGenesis,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CAGO,
            generateTop = False,
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

            numberOfThreads = NUMBER_OF_CPU,

       )
        protGenesisMin.setObjLabel('[GENESIS]\n Energy Minimization C-Alpha Go model')
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        protGenesisFit = self.newProtocol(ProtGenesis,

            inputPDB = protGenesisMin.outputPDBs,
            forcefield = FORCEFIELD_CAGO,
            generateTop = False,
            inputTOP = protGenesisMin.getInputPDBprefix()+".top",
            restartchoice = True,
            inputRST = protGenesisMin.getOutputPrefix()+".rst",

            simulationType = SIMULATION_MD,
            integrator = INTEGRATOR_VVERLET,
            time_step = 0.0005,
            n_steps = 20000,
            eneout_period = 1000,
            crdout_period = 1000,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            ensemble = ENSEMBLE_NVT,
            tpcontrol = TPCONTROL_LANGEVIN,
            temperature = 100.0,

            boundary = BOUNDARY_NOBC,
            EMfitChoice = EMFIT_VOLUMES,
            constantK = 100,
            emfit_sigma = 2.0,
            emfit_tolerance = 0.1,
            inputVolume = protImportVol.outputVolume,
            voxel_size = 2.0,
            centerOrigin = True,

            numberOfThreads = NUMBER_OF_CPU,
        )
        protGenesisFit.setObjLabel('[GENESIS]\n MD cryo-EM fitting with C-Alpha Go model')

        # Launch Fitting
        self.launchProtocol(protGenesisFit)

        # Get GENESIS log file
        log_file = protGenesisFit.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Get the RMSD from the dcd file
        matchingAtoms = matchPDBatoms([PDBMol(protGenesisMin.getInputPDBprefix() + ".pdb")
                                          , PDBMol(self.ds.getFile('1ake_pdb'))])

        rmsd = rmsdFromDCD(outputPrefix = protGenesisFit.getOutputPrefix(),
                           inputPDB = protGenesisFit.getInputPDBprefix()+".pdb",
                           targetPDB= self.ds.getFile('1ake_pdb'),
                           idx=matchingAtoms,
                           align=False)

        # Assert that the CC is increasing
        print("\n\n//////////////////////////////////////////////")
        print(protGenesisFit.getObjLabel())
        print("Initial CC : %.2f"%cc[0])
        print("Final CC : %.2f"%cc[-1])
        print("Initial rmsd : %.2f Ang"%rmsd[0])
        print("Final rmsd : %.2f Ang"%rmsd[-1])
        print("//////////////////////////////////////////////\n\n")
        assert(cc[0] < cc[-1])
        assert(rmsd[0] > rmsd[-1])
        assert(rmsd[-1] < 3.0)

    def testMDCHARMM(self):
        # Import PDB
        protPdbIonize = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                       pdbFile=self.ds.getFile('ionize_pdb'))
        protPdbIonize.setObjLabel('Input PDB (5ftm chain A solvated with water & ions)')
        self.launchProtocol(protPdbIonize)

        # Minimize energy
        protGenesisMin = self.newProtocol(ProtGenesis,
              inputPDB = protPdbIonize.outputPdb,
              forcefield = FORCEFIELD_CHARMM,
              inputPRM = self.ds.getFile('charmm_prm'),
              inputRTF = self.ds.getFile('charmm_top'),
              inputPSF = self.ds.getFile('ionize_psf'),
              inputSTR = self.ds.getFile('charmm_str'),

              md_program = PROGRAM_SPDYN,
              simulationType = SIMULATION_MIN,
              time_step = 0.002,
              n_steps = 100, # 2000
              eneout_period = 10,
              crdout_period = 10,
              nbupdate_period = 10,

              electrostatics = ELECTROSTATICS_PME,
              switch_dist = 10.0,
              cutoff_dist = 12.0,
              pairlist_dist = 15.0,

              boundary = BOUNDARY_PBC,
              box_size_x = 101.4,
              box_size_y = 113.6,
              box_size_z = 81.4,

              rigid_bond = True,
              fast_water = True,
              water_model = "TIP3",

              numberOfThreads=NUMBER_OF_CPU,
          )
        protGenesisMin.setObjLabel("[GENESIS]\n Energy Minimization CHARMM Explicit solvent")
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # Get GENESIS log file
        output_prefix = protGenesisMin.getOutputPrefix()
        log_file = output_prefix + ".log"

        # Get the potential energy from the log file
        potential_ene = readLogFile(log_file)["POTENTIAL_ENE"]

        # Assert that the potential energy is decreasing
        print("\n\n//////////////////////////////////////////////")
        print(protGenesisMin.getObjLabel())
        print("Initial potential energy : %.2f kcal/mol" % potential_ene[0])
        print("Final potential energy : %.2f kcal/mol" % potential_ene[-1])
        print("//////////////////////////////////////////////\n\n")

        assert (potential_ene[0] > potential_ene[-1])

        protGenesisMDRun = self.newProtocol(ProtGenesis,
                    inputPDB=protGenesisMin.outputPDBs,
                    forcefield=FORCEFIELD_CHARMM,
                    inputPRM=self.ds.getFile('charmm_prm'),
                    inputRTF=self.ds.getFile('charmm_top'),
                    inputPSF=self.ds.getFile('ionize_psf'),
                    inputSTR=self.ds.getFile('charmm_str'),
                    restartchoice=True,
                    inputRST=protGenesisMin.getOutputPrefix() + ".rst",

                    md_program=PROGRAM_SPDYN,
                    integrator=INTEGRATOR_VVERLET,
                    time_step=0.002,
                    n_steps=100,
                    eneout_period=100,
                    crdout_period=100,
                    nbupdate_period=10,

                    electrostatics=ELECTROSTATICS_PME,
                    switch_dist=10.0,
                    cutoff_dist=12.0,
                    pairlist_dist=15.0,

                    ensemble=ENSEMBLE_NPT,
                    tpcontrol=TPCONTROL_LANGEVIN,
                    temperature=300.0,
                    pressure=1.0,

                    boundary=BOUNDARY_PBC,
                    box_size_x=101.4,
                    box_size_y=113.6,
                    box_size_z=81.4,

                    rigid_bond=True,
                    fast_water=True,
                    water_model="TIP3",

                    numberOfThreads=NUMBER_OF_CPU,
                                          )
        protGenesisMDRun.setObjLabel("[GENESIS]\n MD simulation with CHARMM explicit solvent")
        # Launch Simulation
        self.launchProtocol(protGenesisMDRun)

