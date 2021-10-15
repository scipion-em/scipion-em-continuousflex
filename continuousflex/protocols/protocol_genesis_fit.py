# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
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


import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from pwem.objects.data import AtomStruct, SetOfAtomStructs, SetOfPDBs, SetOfVolumes

import numpy as np
import mrcfile
import os
from skimage.exposure import match_histograms
from .utilities.pdb_analysis import Molecule, get_mols_conv


class FlexProtGenesisFit(ProtAnalysis3D):
    """ Protocol to use GENESIS. """
    _label = 'Genesis'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        # GENERAL =================================================================================================
        form.addSection(label='General Params')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation', important=True)
        form.addParam('n_proc', params.IntParam, default=1, label='Number of processors',
                      help="TODO")
        form.addParam('n_threads', params.IntParam, default=1, label='Number of threads',
                      help="TODO")

        # Inputs ============================================================================================
        form.addSection(label='Inputs')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct, SetOfPDBs, SetOfAtomStructs', label="Input PDB (s)",
                      help='Select the input PDB or set of PDBs.', important=True)
        form.addParam('forcefield', params.EnumParam, label="Forcefield type", default=0, important=True,
                      choices=['CHARMM', 'AAGO', 'CAGO'], help="TODo")
        form.addParam('generateTop', params.EnumParam, label="Generate topology files ?", default=0,
                      choices=['Yes', 'No'], help="TODo")
        form.addParam('smog_dir', params.FileParam, label="SMOG2 directory",
                      help='TODO', condition="(forcefield==1 or forcefield==2) and generateTop == 0")
        form.addParam('inputGRO', params.FileParam, label="GROMACS Coordinates File (.gro)",
                      condition="(forcefield==1 or forcefield==2) and generateTop==1",
                      help='TODO')
        form.addParam('inputTOP', params.FileParam, label="GROMACS Topology File (.top)",
                      condition="(forcefield==1 or forcefield==2) and generateTop==1",
                      help='TODO')
        form.addParam('inputPRM', params.FileParam, label="CHARMM Parameter File (.prm)",
                      condition = "forcefield==0",
                      help='CHARMM force field parameter file (.prm). Can be founded at ' +
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')
        form.addParam('inputRTF', params.FileParam, label="CHARMM Topology File (.rtf)",
                      condition="forcefield==0 or ((forcefield==1 or forcefield==2) and generateTop == 0)",
                      help='CHARMM force field topology file (.rtf). Can be founded at ' +
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm. '+
                           'In the case of AAGO/CAGO model, used for completing the missing structure')


        form.addParam('inputPSF', params.FileParam, label="Protein Structure File (.psf)",
                      condition="forcefield==0 and generateTop==1",
                      help='TODO')

        form.addParam('restartchoice', params.EnumParam, label="Restart previous run", default=0,
                      choices=['Yes', 'No'],help="TODo")
        form.addParam('inputRST', params.FileParam, label="GENESIS Restart File (.rst)",
                       help='Restart file from previous minimisation or MD run '
                      , condition="restartchoice==0")


        # Simulation =================================================================================================
        form.addSection(label='Simulation')
        form.addParam('simulationType', params.EnumParam, label="Simulation type", default=0,
                      choices=['Molecular Dynamics', 'Minimization'],  help="TODO", important=True)
        form.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['Velocity Verlet', 'Leapfrog'],  help="TODO", condition="simulationType==0")
        form.addParam('time_step', params.FloatParam, default=0.002, label='Time step (ps)',
                      help="TODO", condition="simulationType==0")
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Select the number of steps in the MD fitting")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="TODO")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinates output period',
                      help="TODO")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="TODO")
        # ENERGY =================================================================================================
        form.addSection(label='Energy')
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['GBSA', 'NONE'],
                      help="TODo")
        form.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance', help="TODO")
        form.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance', help="TODO")
        form.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance', help="TODO")
        form.addParam('tpcontrol', params.EnumParam, label="Temperature control", default=0,
                      choices=['LANGEVIN', 'BERENDSEN', 'NO'],
                      help="TODo")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="TODO")
        # EM fit =================================================================================================
        form.addSection(label='EM fit')
        form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=1,
                      choices=['Yes', 'No'], important=True,
                      help="TODO")
        form.addParam('constantK', params.IntParam, default=10000, label='Force constant K',
                      help="TODO", condition="EMfitChoice==0")
        form.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EMfit Sigma",
                      help="TODO", condition="EMfitChoice==0")
        form.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EMfit Tolerance',
                      help="TODO", condition="EMfitChoice==0")
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume, SetOfVolumes",
                      label="Input volume (s)", help='Select the target EM density volume', condition="EMfitChoice==0")
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="TODO", condition="EMfitChoice==0")
        form.addParam('situs_dir', params.FileParam,
                      label="Situs install path", help='Select the root directory of Situs installation'
                      , condition="EMfitChoice==0")
        form.addParam('centerOrigin', params.EnumParam, label="Center Origin", default=0,
                      choices=['Yes', 'NO'],
                      help="TODo", condition="EMfitChoice==0")
        # NMMD =================================================================================================
        form.addSection(label='NMMD')
        form.addParam('normalModesChoice', params.EnumParam, label="Normal Mode Molecular Dynamics", default=1,
                      choices=['Yes', 'No'], important=True,
                      help="TODO")
        form.addParam('n_modes', params.IntParam, default=10, label='Number of normal modes',
                      help="TODO", condition="normalModesChoice==0")
        form.addParam('global_mass', params.FloatParam, default=1.0, label='Normal modes amplitude mass',
                      help="TODO", condition="normalModesChoice==0")
        form.addParam('global_limit', params.FloatParam, default=300.0, label='Normal mode amplitude threshold',
                      help="TODO", condition="normalModesChoice==0")
        # REMD =================================================================================================
        form.addSection(label='REMD')
        form.addParam('replica_exchange', params.EnumParam, label="Replica Exchange", default=1,
                      choices=['Yes', 'No'], important=True,
                      help="TODO")
        form.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="TODO", condition="replica_exchange==0")
        form.addParam('nreplica', params.IntParam, default=1, label='Number of replicas',
                      help="TODO", condition="replica_exchange==0")
        form.addParam('constantKREMD', params.StringParam, label='K values ',
                      help="TODO", condition="replica_exchange==0")
        # Outputs =================================================================================================
        form.addSection(label='Outputs')
        form.addParam('rmsdChoice', params.EnumParam, label="RMSD to target PDB", default=1,
                      choices=['Yes', 'No'], important=False,
                      help="TODO")
        form.addParam('target_pdb', params.PointerParam,
                      pointerClass='AtomStruct', label="Target PDB", help='TODO', condition="rmsdChoice==0")

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        self._insertFunctionStep("createINPStep")
        self._insertFunctionStep("fittingStep")
        self._insertFunctionStep("createOutputStep")

    def createINPStep(self):
        # CREATE INPUT FILE FOR GENESIS
        for i in range(self.numberOfFitting):
            outputPrefix = self._getExtraPath("%s_output" % (str(i+1).zfill(3)))

            s = "\n[INPUT] \n"
            if self.forcefield.get() == 0:
                s += "topfile = %s\n" % self.inputRTF.get()
                s += "parfile = %s\n" % self.inputPRM.get()
                if self.numberOfInputPDB == 1:
                    s += "pdbfile = %s\n" % self.inputPDBfn[0]
                    s += "psffile = %s\n" % self.inputPSFfn[0]
                else:
                    s += "pdbfile = %s\n" % self.inputPDBfn[i]
                    s += "psffile = %s\n" % self.inputPSFfn[i]
            elif self.forcefield.get() == 1 or self.forcefield.get() == 2:
                if self.numberOfInputPDB == 1:
                    if len(self.inputGROfn)==0:
                        s += "pdbfile = %s\n" % self.inputPDBfn[0]
                    else:
                        s += "grotopfile = %s\n" % self.inputTOPfn[0]
                    s += "grocrdfile = %s \n" % self.inputGROfn[0]
                else:
                    if len(self.inputGROfn) == 0:
                        s += "pdbfile = %s\n" % self.inputPDBfn[i]
                    else:
                        s += "grotopfile = %s\n" % self.inputTOPfn[i]
                    s += "grocrdfile = %s \n" % self.inputGROfn[i]
            if self.restartchoice.get() == 0:
                s += "rstfile = %s\n" % self.inputRST.get()

            s += "\n[OUTPUT] \n"
            if self.replica_exchange.get() == 0:
                outputPrefix += "_remd{}"
                s += "remfile = %s.rem\n" %outputPrefix
                s += "logfile = %s.log\n" %outputPrefix
            s += "dcdfile = %s.dcd\n" %outputPrefix
            s += "rstfile = %s.rst\n" %outputPrefix
            s += "pdbfile = %s.pdb\n" %outputPrefix

            s += "\n[ENERGY] \n"
            if self.forcefield.get() == 0:
                s += "forcefield = CHARMM \n"
            elif self.forcefield.get() == 1:
                s += "forcefield = AAGO  \n"
            elif self.forcefield.get() == 2:
                s += "forcefield = CAGO  \n"
            s += "electrostatic = CUTOFF  \n"
            s += "switchdist   = %.2f \n" % self.switch_dist.get()
            s += "cutoffdist   = %.2f \n" % self.cutoff_dist.get()
            s += "pairlistdist = %.2f \n" % self.pairlist_dist.get()
            s += "vdw_force_switch = YES \n"
            if self.implicitSolvent.get() == 0:
                s += "implicit_solvent = GBSA \n"
                s += "gbsa_eps_solvent = 78.5 \n"
                s += "gbsa_eps_solute  = 1.0 \n"
                s += "gbsa_salt_cons   = 0.2 \n"
                s += "gbsa_surf_tens   = 0.005 \n"
            else:
                s += "implicit_solvent = NONE  \n"

            if self.simulationType.get() == 1:
                s += "\n[MINIMIZE]\n"
                s += "method = SD\n"
            else:
                s += "\n[DYNAMICS] \n"
                if self.integrator.get() == 0:
                    s += "integrator = VVER  \n"
                else:
                    s += "integrator = LEAP  \n"
                s += "timestep = %f \n" % self.time_step.get()
            s += "nsteps = %i \n" % self.n_steps.get()
            s += "eneout_period = %i \n" % self.eneout_period.get()
            s += "crdout_period = %i \n" % self.crdout_period.get()
            s += "rstout_period = %i \n" % self.n_steps.get()
            s += "nbupdate_period = %i \n" % self.nbupdate_period.get()

            s += "\n[CONSTRAINTS] \n"
            s += "rigid_bond = NO \n"

            s += "\n[ENSEMBLE] \n"
            s += "ensemble = NVT \n"
            if self.tpcontrol.get() == 0:
                s += "tpcontrol = LANGEVIN  \n"
            elif self.tpcontrol.get() == 1:
                s += "tpcontrol = BERENDSEN  \n"
            else:
                s += "tpcontrol = NO  \n"
            s += "temperature = %.2f \n" % self.temperature.get()

            s += "\n[BOUNDARY] \n"
            s += "type = NOBC  \n"

            if self.EMfitChoice.get()==0 and self.simulationType.get() == 0:
                s += "\n[SELECTION] \n"
                s += "group1 = all and not hydrogen\n"

                s += "\n[RESTRAINTS] \n"
                s += "nfunctions = 1 \n"
                s += "function1 = EM \n"
                if self.replica_exchange.get() == 0:
                    s += "constant1 = %s \n" % self.constantKREMD.get()
                else:
                    s += "constant1 = %.2f \n" % self.constantK.get()
                s += "select_index1 = 1 \n"

                s += "\n[EXPERIMENTS] \n"
                s += "emfit = YES  \n"
                if self.numberOfInputVol == 1 :
                    s += "emfit_target = %s \n" % self.inputVolumefn[0]
                else:
                    s += "emfit_target = %s \n" % self.inputVolumefn[i]
                s += "emfit_sigma = %.4f \n" % self.emfit_sigma.get()
                s += "emfit_tolerance = %.6f \n" % self.emfit_tolerance.get()
                s += "emfit_period = 1  \n"

                if self.replica_exchange.get() == 0:
                    s += "\n[REMD] \n"
                    s += "dimension = 1 \n"
                    s += "exchange_period = %i \n" % self.exchange_period.get()
                    s += "type1 = RESTRAINT \n"
                    s += "nreplica1 = %i \n" % self.nreplica.get()
                    s += "rest_function1 = 1 \n"

            with open(self._getExtraPath("%s_INP"% str(i+1).zfill(3)), "w") as f:
                f.write(s)

    def fittingStep(self):
        # RUN GENESIS FOR EACH INP FILE
        for i in range(self.numberOfFitting):
            outputPrefix = self._getExtraPath("%s_output" % (str(i+1).zfill(3)))
            with open(self._getExtraPath("launch_genesis.sh"), "w") as f:
                f.write("export OMP_NUM_THREADS=%i\n"% self.n_threads.get())
                if (self.n_proc.get() != 1) :
                    f.write("mpirun -np %s " %self.n_proc.get())
                f.write("%s/bin/atdyn %s " %
                        (self.genesisDir.get(),
                         self._getExtraPath("%s_INP"% str(i+1).zfill(3))))
                if self.normalModesChoice.get() == 0:
                    f.write("%s/ %i %f %f" % (self.genesisDir.get(),
                             self.n_modes.get(),
                             self.global_mass.get(),
                             self.global_limit.get()))
                if self.replica_exchange.get() == 1:
                        f.write(" | tee %s.log"%outputPrefix)
                f.write("\nexit")
            self.runJob("chmod", "777 "+self._getExtraPath("launch_genesis.sh"))
            self.runJob(self._getExtraPath("launch_genesis.sh"), "")

            # COMPUTE CC AND RMSD IF NEEDED
            if self.EMfitChoice.get()==0:
                for j in range(self.numberOfReplicas):
                    if self.replica_exchange.get() == 0 :
                        outputPrefix = self._getExtraPath("%s_output_remd%i" % (str(i+1).zfill(3), j+1))

                    # comp CC
                    self.ccFromLogFile(outputPrefix)

                    # comp RMSD
                    if self.rmsdChoice.get() == 0:
                        inputPDB = self.inputPDBfn[0] \
                            if self.numberOfInputPDB == 1 else self.inputPDBfn[i]
                        self.rmsdFromDCD(outputPrefix, inputPDB)


    def convertInputStep(self):

        # SETUP INPUT PDBs
        initFn = []
        if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                isinstance(self.inputPDB.get(), SetOfPDBs):
            self.numberOfInputPDB = self.inputPDB.get().getSize()
            for i in range(self.inputPDB.get().getSize()):
                initFn.append(self.inputPDB.get()[i+1].getFileName())

        else:
            self.numberOfInputPDB =1
            initFn.append(self.inputPDB.get().getFileName())

        # COPY INIT PDBs
        self.inputPDBfn = []
        for i in range(self.numberOfInputPDB):
            newPDB = self._getExtraPath("%s_inputPDB.pdb" % str(i + 1).zfill(3))
            self.inputPDBfn.append(newPDB)
            os.system("cp %s %s"%(initFn[i], newPDB))

        # GENERATE TOPOLOGY FILES
        if self.generateTop.get()==0:

            #CHARMM
            if self.forcefield.get() == 0:
                self.inputPSFfn = []
                for i in range(self.numberOfInputPDB):
                    inputPrefix = self._getExtraPath("%s_inputPDB"%str(i+1).zfill(3))
                    self.generatePSF(self.inputPDBfn[i],inputPrefix)
                    self.inputPSFfn.append(inputPrefix+".psf")

            # GROMACS
            elif self.forcefield.get() == 1 or self.forcefield.get() == 2:
                self.inputGROfn = []
                self.inputTOPfn = []
                for i in range(self.numberOfInputPDB):
                    inputPrefix = self._getExtraPath("%s_inputPDB" % str(i + 1).zfill(3))
                    self.generatePSF(self.inputPDBfn[i], inputPrefix)
                    self.generateGROTOP(self.inputPDBfn[i], inputPrefix)
                    self.inputGROfn.append(inputPrefix+".gro")
                    self.inputTOPfn.append(inputPrefix+".top")

        else:
            if self.forcefield.get() == 0:
                self.inputPSFfn = [self.inputPSF.get()]

            elif self.forcefield.get() == 1 or self.forcefield.get() == 2:
                if self.inputGRO.get() == "":
                    self.inputGROfn = []
                else:
                    self.inputGROfn = [self.inputGRO.get()]
                self.inputTOPfn = [self.inputTOP.get()]

        # SETUP INPUT VOLUMES
        if self.EMfitChoice.get()==0:
            self.inputVolumefn = []
            if isinstance(self.inputVolume.get(), SetOfVolumes) :
                self.numberOfInputVol = self.inputVolume.get().getSize()
                for i in self.inputVolume.get():
                    self.inputVolumefn.append(i.getFileName())
            else:
                self.numberOfInputVol =1
                self.inputVolumefn.append(self.inputVolume.get().getFileName())

            if self.numberOfInputPDB != self.numberOfInputVol and \
                    self.numberOfInputVol != 1 and self.numberOfInputPDB != 1:
                raise RuntimeError("Number of input volumes and PDBs must be the same.")

            # CONVERT VOLUMES
            for i in range(self.numberOfInputVol):
                volPrefix = self._getExtraPath("%s_inputVol" % str(i+1).zfill(3))
                fnPDB = self.inputPDBfn[0] if self.numberOfInputPDB == 1 else self.inputPDBfn[i]

                self.inputVolumefn[i]  = self.convertVol(fnInput=self.inputVolumefn[i],
                                   volPrefix = volPrefix, fnPDB=fnPDB)
        else:
            self.numberOfInputVol=0

        self.numberOfReplicas = self.nreplica.get() if self.replica_exchange.get() == 0 else 1
        self.numberOfFitting = np.max([self.numberOfInputPDB,self.numberOfInputVol])


    def createOutputStep(self):

        # CREATE SET OF PDBs
        pdbset = self._createSetOfPDBs("outputPDBs")
        for i in range(self.numberOfFitting):
            outputPrefix = self._getExtraPath("%s_output" % str(i + 1).zfill(3))
            for j in range(self.numberOfReplicas):
                if self.replica_exchange.get() == 0:
                    outputPrefix = self._getExtraPath("%s_output_remd%i" % (str(i + 1).zfill(3), j + 1))
                pdbset.append(AtomStruct(outputPrefix + ".pdb"))

        self._defineOutputs(outputPDBs=pdbset)

    def convertVol(self,fnInput,volPrefix, fnPDB):

        # CONVERT TO MRC
        pre, ext = os.path.splitext(os.path.basename(fnInput))
        if ext != ".mrc":
            self.runJob("xmipp_image_convert", "-i %s --oext mrc -o %s.mrc" %
                        (fnInput,volPrefix))
        else:
            os.system("cp %s %s.mrc" %(fnInput,volPrefix))

        # READ INPUT MRC
        with mrcfile.open("%s.mrc" % volPrefix) as input_mrc:
            inputMRCData = input_mrc.data
            inputMRCShape = inputMRCData.shape
            if self.centerOrigin.get() == 0 :
                origin = -self.voxel_size.get() * (np.array(inputMRCData.shape)) / 2
            else:
                origin = np.zeros(3)

        # CONVERT PDB TO SITUS VOLUME USING EMMAP GENERATOR
        fnTmpVol = self._getExtraPath("tmp")
        s ="\n[INPUT] \n"
        s +="pdbfile = %s\n" % fnPDB
        s +="\n[OUTPUT] \n"
        s +="mapfile = %s.sit\n" % fnTmpVol
        s +="\n[OPTION] \n"
        s +="map_format = SITUS \n"
        s +="voxel_size = %f \n" % self.voxel_size.get()
        s +="sigma = %f  \n" % self.emfit_sigma.get()
        s +="tolerance = %f  \n"% self.emfit_tolerance.get()
        s +="auto_margin    = NO\n"
        s +="x0             = %f \n" % origin[0]
        s +="y0             = %f \n" % origin[1]
        s +="z0             = %f \n" % origin[2]
        s +="box_size_x     =  %f \n" % (inputMRCShape[0]*self.voxel_size.get())
        s +="box_size_y     =  %f \n" % (inputMRCShape[1]*self.voxel_size.get())
        s +="box_size_z     =  %f \n" % (inputMRCShape[2]*self.voxel_size.get())
        with open("%s_INP_emmap" % fnTmpVol, "w") as f:
            f.write(s)
        self.runJob("%s/bin/emmap_generator" % self.genesisDir.get(), "%s_INP_emmap" % fnTmpVol)

        # CONVERT SITUS TMP FILE TO MRC
        with open(self._getExtraPath("runconvert.sh"), "w") as f:
            f.write("#!/bin/bash \n")
            f.write("%s/bin/map2map %s %s <<< \'1\'\n" % (self.situs_dir.get(), fnTmpVol+".sit", fnTmpVol+".mrc"))
            f.write("exit")
        self.runJob("/bin/bash", self._getExtraPath("runconvert.sh"))

        # READ GENERATED MRC
        with mrcfile.open(fnTmpVol+".mrc") as tmp_mrc:
            tmpMRCData = tmp_mrc.data

        # MATCH HISTOGRAMS
        mrc_data = match_histograms(inputMRCData, tmpMRCData)

        # SAVE TO MRC
        with mrcfile.new("%sConv.mrc"%volPrefix, overwrite=True) as mrc:
            mrc.set_data(np.float32(mrc_data))
            mrc.voxel_size = self.voxel_size.get()
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
            mrc.update_header_from_data()
            mrc.update_header_stats()

        # CONVERT MRC TO SITUS
        with open(self._getExtraPath("runconvert.sh"), "w") as f:
            f.write("#!/bin/bash \n")
            f.write("%s/bin/map2map %s %s <<< \'1\'\n" % (self.situs_dir.get(),
                                                          "%sConv.mrc"%volPrefix, "%s.sit"%volPrefix))
            f.write("exit")
        self.runJob("/bin/bash", self._getExtraPath("runconvert.sh"))

        # CLEANING
        os.system("rm -f %s.sit"%fnTmpVol)
        os.system("rm -f %s.mrc"%fnTmpVol)
        os.system("rm -f %s"%self._getExtraPath("runconvert.sh"))
        os.system("rm -f %s_INP_emmap" % fnTmpVol)
        os.system("rm -f %sConv.mrc"%volPrefix)
        os.system("rm -f %s.mrc" % volPrefix)

        return "%s.sit"%volPrefix

    def generatePSF(self, inputPDB, outputPrefix):

        fnPSFgen = self._getExtraPath("psfgen.tcl")
        with open(fnPSFgen, "w") as psfgen:
            psfgen.write("mol load pdb %s\n" % inputPDB)
            psfgen.write("\n")
            psfgen.write("package require psfgen\n")
            psfgen.write("topology %s\n" % self.inputRTF.get())
            psfgen.write("pdbalias residue HIS HSE\n")
            psfgen.write("pdbalias residue MSE MET\n")
            psfgen.write("pdbalias atom ILE CD1 CD\n")
            psfgen.write("pdbalias residue A ADE\n")
            psfgen.write("pdbalias residue G GUA\n")
            psfgen.write("pdbalias residue C CYT\n")
            psfgen.write("pdbalias residue T THY\n")
            psfgen.write("pdbalias residue U URA\n")
            psfgen.write("\n")
            psfgen.write("set sel [atomselect top nucleic]\n")
            psfgen.write("set chains [lsort -unique [$sel get chain]] ;\n")
            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set seg ${chain}DNA\n")
            psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
            psfgen.write("    $sel set segid $seg\n")
            psfgen.write("    $sel writepdb tmp.pdb\n")
            psfgen.write("    segment $seg { pdb tmp.pdb }\n")
            psfgen.write("    coordpdb tmp.pdb\n")
            psfgen.write("}\n")
            psfgen.write("\n")
            psfgen.write("set protein [atomselect top protein]\n")
            psfgen.write("puts \"///////////////test///////////////\"\n")
            psfgen.write("puts $protein\n")
            psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
            psfgen.write("puts \"///////////////test///////////////\"\n")
            psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set sel [atomselect top \"pfrag $chain\"]\n")
            psfgen.write("    $sel writepdb tmp.pdb\n")
            psfgen.write("    segment U${chain} {pdb tmp.pdb}\n")
            psfgen.write("    coordpdb tmp.pdb U${chain}\n")
            psfgen.write("}\n")
            psfgen.write("rm -f tmp.pdb\n")
            psfgen.write("\n")
            psfgen.write("guesscoord\n")
            psfgen.write("writepdb %s.pdb\n" % outputPrefix)
            psfgen.write("writepsf %s.psf\n" % outputPrefix)
            psfgen.write("exit\n")
        self.runJob("vmd", "-dispdev text -e "+fnPSFgen)

    def generateGROTOP(self,inputPDB, inputPrefix):
        mol = Molecule(inputPDB)
        mol.remove_alter_atom()
        mol.remove_hydrogens()
        mol.alias_atom("CD", "CD1", "ILE")
        mol.alias_atom("OT1", "O")
        mol.alias_atom("OT2", "OXT")
        mol.alias_res("HSE", "HIS")
        mol.alias_res("CYT", "C")
        mol.alias_res("GUA", "G")
        mol.alias_res("ADE", "A")
        mol.alias_res("URA", "U")
        mol.alias_atom("O1'", "O1*")
        mol.alias_atom("O2'", "O2*")
        mol.alias_atom("O3'", "O3*")
        mol.alias_atom("O4'", "O4*")
        mol.alias_atom("O5'", "O5*")
        mol.alias_atom("C1'", "C1*")
        mol.alias_atom("C2'", "C2*")
        mol.alias_atom("C3'", "C3*")
        mol.alias_atom("C4'", "C4*")
        mol.alias_atom("C5'", "C5*")
        mol.add_terminal_res()
        mol.atom_res_reorder()
        mol.save_pdb(inputPDB)

        # Run Smog2
        self.runJob("%s/bin/smog2" % self.smog_dir.get(),
                    "-i %s -dname %s -%s -limitbondlength -limitcontactlength" %
                    (inputPDB, inputPrefix, "CA" if self.forcefield.get() == 2 else "AA"))

        # ADD CHARGE TO FILE
        grotopFile = inputPrefix + ".top"
        with open(grotopFile, 'r') as f1:
            with open(grotopFile+".tmp", 'w') as f2:
                atom_scope = False
                write_line = False
                for line in f1:
                    if "[" in line and "]" in line:
                        if "atoms" in line:
                            atom_scope = True
                    if atom_scope:
                        if "[" in line and "]" in line:
                            if not "atoms" in line:
                                atom_scope = False
                                write_line = False
                        elif not ";" in line and not (not line or line.isspace()):
                            write_line = True
                        else:
                            write_line = False
                    if write_line:
                        f2.write("%s\t0.0\n" % line[:-1])
                    else:
                        f2.write(line)
        os.system("cp %s.tmp %s ; rm -f %s.tmp"%(grotopFile,grotopFile,grotopFile))

        # SELECT CA ATOMS IF CAGO MODEL
        if self.forcefield.get() == 2 :
            initPDB = Molecule(inputPDB)
            initPDB.allatoms2ca()
            initPDB.save_pdb(inputPDB)

    def rmsdFromDCD(self, outputPrefix, inputPDB):

        # EXTRACT PDBs from dcd file
        with open("%s_dcd2pdb.tcl" % outputPrefix, "w") as f:
            s = ""
            s += "mol load pdb %s dcd %s.dcd\n" % (inputPDB, outputPrefix)
            s += "set nf [molinfo top get numframes]\n"
            s += "for {set i 0 } {$i < $nf} {incr i} {\n"
            s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" % outputPrefix
            s += "}\n"
            s += "exit\n"
            f.write(s)
        os.system("vmd -dispdev text -e %s_dcd2pdb.tcl > /dev/null" % outputPrefix)

        # DEF RMSD
        def RMSD(c1, c2):
            return np.sqrt(np.mean(np.square(np.linalg.norm(c1 - c2, axis=1))))

        # COMPUTE RMSD
        rmsd = []
        N = (self.n_steps.get() // self.crdout_period.get())
        initPDB = Molecule(inputPDB)
        targetPDB = Molecule(self.target_pdb.get().getFileName())

        idx = get_mols_conv([initPDB, targetPDB], ca_only=True)
        if len(idx) > 0:
            rmsd.append(RMSD(initPDB.coords[idx[:, 0]], targetPDB.coords[idx[:, 1]]))
            for i in range(N):
                mol = Molecule(outputPrefix + "tmp" + str(i + 1) + ".pdb")
                rmsd.append(RMSD(mol.coords[idx[:, 0]], targetPDB.coords[idx[:, 1]]))
        else:
            rmsd = np.zeros(N + 1)

        # CLEAN TMP FILES AND SAVE
        os.system("rm -f %stmp*" % (outputPrefix))
        np.savetxt(outputPrefix + "_rmsd.txt", rmsd)

    def ccFromLogFile(self,outputPrefix):
        # READ CC IN GENESIS LOG FILE
        with open(outputPrefix+".log","r") as f:
            header = None
            cc = []
            cc_idx = 0
            for i in f:
                if i.startswith("INFO:"):
                    if header is None:
                        header = i.split()
                        for i in range(len(header)):
                            if 'RESTR_CVS001' in header[i]:
                                cc_idx = i
                    else:
                        splitline = i.split()
                        if len(splitline) == len(header):
                            cc.append(float(splitline[cc_idx]))

        # SAVE
        np.savetxt(outputPrefix +"_cc.txt", np.array(cc))


    # --------------------------- STEPS functions --------------------------------------------
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
