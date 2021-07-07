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
from pwem.objects.data import AtomStruct
import numpy as np

import sys
sys.path.append('/home/guest/PycharmProjects/bayesian-md-nma')
from src.molecule import Molecule
from src.functions import get_cc_rmsd
import matplotlib.pyplot as plt
import time
import mrcfile

import os

TPCONTROL_LANGEVIN=0
TPCONTROL_BERENDSEN=1
TPCONTROL_NO=2

class FlexProtGenesisFit(ProtAnalysis3D):
    """ Protocol for flexible fitting a PDB into a cryo EM map using Genesis. """
    _label = 'genesis fit'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        # GENERAL =================================================================================================
        form.addSection(label='Input')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation')
        form.addParam('forcefield', params.EnumParam, label="Forcefield", default=0,
                      choices=['CHARMM', 'AAGO'],help="TODo")
        form.addParam('inputGenesisMin', params.PointerParam,
                      pointerClass='FlexProtGenesisMin', label="Input Genesis Minimization",
                      help='Select the input minimization', condition="forcefield==0")
        form.addParam('inputGRO', params.PointerParam, label="Coordinate file .gro",
                      pointerClass='EMFile', help='TODO', condition="forcefield==1")
        form.addParam('inputTOP', params.PointerParam, label="Topology file .top",
                      pointerClass='EMFile', help='TODO', condition="forcefield==1")
        form.addParam('n_proc', params.IntParam, default=1, label='Number of processors',
                      help="TODO")
        form.addParam('n_threads', params.IntParam, default=1, label='Number of threads',
                      help="TODO")
        form.addParam('molprobity', params.EnumParam, label="Do Molprobity ?", default=0,
                      choices=['Yes', 'No'],
                      help="TODO")

        # ENERGY =================================================================================================
        form.addSection(label='Energy')
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['Yes', 'NO'],
                      help="TODo")
        form.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance', help="TODO")
        form.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance', help="TODO")
        form.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance', help="TODO")

        # DYNAMICS =================================================================================================
        form.addSection(label='Dynamics')
        form.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['VVER', 'LEAP'],  help="TODO")
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Select the number of steps in the MD fitting")
        form.addParam('time_step', params.FloatParam, default=0.001, label='Time step (ps)',
                      help="TODO")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="TODO")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinates output period',
                      help="TODO")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="TODO")
        # Constraints =================================================================================================
        # form.addSection(label='Constraints')

        # Ensemble =================================================================================================
        form.addSection(label='Ensemble')
        form.addParam('tpcontrol', params.EnumParam, label="Temperature control", default=0,
                      choices=['LANGEVIN', 'BERENDSEN', 'NO'],
                      help="TODo")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="TODO")
        # Boundary =================================================================================================
        # form.addSection(label='Boundary')

        # Experiments =================================================================================================
        form.addSection(label='Experiments')
        form.addParam('constantK', params.IntParam, default=10000, label='Force constant K',
                      help="TODO")
        form.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EMfit Sigma",
                      help="TODO")
        form.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EMfit Tolerance',
                      help="TODO")
        form.addParam('convertVolume', params.BooleanParam, label="Convert Volume (SITUS)",
                      default=False,
                      help="If selected, the input MRC volume will be automatically converted to SITUS format")
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume", condition="convertVolume==True",
                      label="Input volume", help='Select the target EM density volume')
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="TODO")
        form.addParam('situs_dir', params.FileParam, condition="convertVolume==True",
                      label="Situs install path", help='Select the root directory of Situs installation')
        form.addParam('inputVolumeFile', params.FileParam, condition="convertVolume==False",
                      label="Input volume", help='Select the SITUS (.sit) volume file')
        form.addParam('centerOrigin', params.EnumParam, label="Center Origin", default=0,
                      choices=['Yes', 'NO'],
                      help="TODo")
        form.addParam('target_pdb', params.PointerParam,
                      pointerClass='AtomStruct', label="[EXP] Target PDB", help='EXP')
        # Normal modes =================================================================================================
        form.addSection(label='Normal Modes')

        form.addParam('fitGlobal', params.EnumParam, label="Fit Global ?", default=0,
                      choices=['Yes', 'No'],
                      help="TODo")
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of normal mode vectors computed by normal mode analysis.')
        form.addParam('n_modes', params.IntParam, default=3, label='Number of modes',
                      help="TODO")
        form.addParam('first_mode', params.IntParam, default=7, label='First mode',
                      help="TODO")
        form.addParam('global_dt', params.FloatParam, default=10.0, label='Global dt',
                      help="TODO")
        # REMD =================================================================================================
        form.addSection(label='REMD')
        form.addParam('replica_exchange', params.EnumParam, label="Do REMD ?", default=1,
                      choices=['Yes', 'No'],
                      help="TODO")
        form.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="TODO")
        form.addParam('nreplica', params.IntParam, default=4, label='Number of replicas',
                      help="TODO")
        form.addParam('constantKREMD', params.StringParam, label='K values ',
                      help="TODO")

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("createInputStep")
        self._insertFunctionStep("fittingStep")
        self._insertFunctionStep("createOutputStep")

    def fittingStep(self):
        min = self.inputGenesisMin.get()

        s = "\n[INPUT] \n"
        if self.forcefield.get() == 0:
            s += "topfile = "+min.inputRTF.get()+"\n"
            s += "parfile = "+min.inputPRM.get()+"\n"
            s += "psffile = "+min.inputPSF.get().getFileName()+"\n"
            s += "pdbfile = "+min.inputPDB.get().getFileName()+"\n"
            s += "rstfile = "+min._getExtraPath("min.rst")+"\n"

        elif self.forcefield.get() == 1:
            s += "grotopfile = " + self.inputTOP.get().getFileName() + "\n"
            s += "grocrdfile = " + self.inputGRO.get().getFileName() + "\n"

        s += "\n[OUTPUT] \n"
        if self.replica_exchange.get() == 0:
            outputPrefix = self._getExtraPath("run_r{}")
            s += "remfile = " + outputPrefix + ".rem\n"
            s += "logfile = " + outputPrefix + ".log\n"
        else:
            outputPrefix = self._getExtraPath("run_r1")
        s += "dcdfile = " + outputPrefix + ".dcd\n"
        s += "rstfile = " + outputPrefix + ".rst\n"
        s += "pdbfile = " + outputPrefix + ".pdb\n"

        s += "\n[ENERGY] \n"
        if self.forcefield.get() == 0:
            s += "forcefield = CHARMM  # CHARMM force field\n"
        elif self.forcefield.get() == 1:
            s += "forcefield = AAGO  # AAGO\n"
        s += "electrostatic = CUTOFF  # use cutoff scheme for non-bonded terms \n"
        s += "switchdist   = "+str(self.switch_dist.get())+" \n"
        s += "cutoffdist   = "+str(self.cutoff_dist.get())+" \n"
        s += "pairlistdist = "+str(self.pairlist_dist.get())+" \n"
        s += "vdw_force_switch = YES \n"
        if self.implicitSolvent.get() == 0:
            s += "implicit_solvent = GBSA    # [GBSA] \n"
            s += "gbsa_eps_solvent = 78.5    # solvent dielectric constant in GB \n"
            s += "gbsa_eps_solute  = 1.0     # solute dielectric constant in GB \n"
            s += "gbsa_salt_cons   = 0.2     # salt concentration (mol/L) in GB \n"
            s += "gbsa_surf_tens   = 0.005   # surface tension (kcal/mol/A^2) in SA \n"
        else:
            s += "implicit_solvent = NONE    # [None] \n"

        s += "\n[DYNAMICS] \n"
        if self.integrator.get() == 0:
            s += "integrator = VVER  \n"
        else:
            s += "integrator = LEAP  \n"
        s += "nsteps = "+str(self.n_steps.get())+" \n"
        s += "timestep = "+str(self.time_step.get())+"  #\n"
        s += "eneout_period = "+str(self.eneout_period.get())+" \n"
        s += "crdout_period = "+str(self.crdout_period.get())+" \n"
        s += "rstout_period = "+str(self.n_steps.get())+"\n"
        s += "nbupdate_period = "+str(self.nbupdate_period.get())+"\n"
        s += "iseed = "+str(np.random.randint(1, 31415))+"  # random number seed  \n"

        s += "\n[CONSTRAINTS] \n"
        s += "rigid_bond = NO  # use SHAKE \n"

        s += "\n[ENSEMBLE] \n"
        s += "ensemble = NVT  # constant temperature \n"
        if self.tpcontrol.get() == TPCONTROL_LANGEVIN:
            s += "tpcontrol = LANGEVIN  \n"
        elif self.tpcontrol.get() == TPCONTROL_BERENDSEN:
            s += "tpcontrol = BERENDSEN  \n"
        else:
            s += "tpcontrol = NO  \n"
        s += "temperature = "+str(self.temperature.get())+" \n"

        s += "\n[BOUNDARY] \n"
        s += "type = NOBC  # No periodic boundary condition \n"

        s += "\n[SELECTION] \n"
        s += "group1 = all and not hydrogen\n"

        s += "\n[RESTRAINTS] \n"
        s += "nfunctions = 1 \n"
        s += "function1 = EM  # apply restraints from EM density map \n"
        if self.replica_exchange.get() == 0 :
            s += "constant1 = "+self.constantKREMD.get()+" \n"
        else:
            s += "constant1 = "+str(self.constantK.get())+" \n"

        s += "select_index1 = 1  # apply restraint force on protein heavy atoms \n"

        s += "\n[EXPERIMENTS] \n"
        s += "emfit = YES  # perform EM flexible fitting \n"
        s += "emfit_target = "+self.inputVolumeFn+"\n"
        s += "emfit_sigma = "+str(self.emfit_sigma.get())+" \n"
        s += "emfit_tolerance = "+str(self.emfit_tolerance.get())+" \n"
        s += "emfit_period = 1  # emfit force update period \n"
        s += "emfit_nma = "+self._getExtraPath("emfit_nma")+"\n"

        if self.replica_exchange.get() == 0:
            s += "\n[REMD] \n"
            s += "dimension = 1 \n"
            s += "exchange_period = "+ str(self.exchange_period.get())+"\n"
            s += "type1 = RESTRAINT \n"
            s += "nreplica1 = "+ str(self.nreplica.get())+" \n"
            s += "rest_function1 = 1 \n"


        with open(self._getExtraPath("fitting"), "w") as f:
            f.write(s)

        with open(self._getExtraPath("emfit_nma"), "w") as f:
            f.write(str(self.fitGlobal.get())+"\n")
            f.write(os.path.splitext(self.inputModes.get().getFileName())[0]+"/vec.\n")
            f.write(str(self.n_modes.get())+"\n")
            f.write(str(self.first_mode.get())+"\n")
            f.write(str(self.global_dt.get())+"\n")

        with open(self._getExtraPath("launch_genesis.sh"), "w") as f:
            f.write("export OMP_NUM_THREADS="+str(self.n_threads.get())+"\n")
            f.write("echo \"OMP NUM THREADS : \"\n")
            f.write("echo $OMP_NUM_THREADS\n")
            f.write("mpirun -np %s %s/bin/atdyn %s %s\n" %
                    (self.n_proc.get(),self.genesisDir.get(),self._getExtraPath("fitting"),
                     " > "+self._getExtraPath("run_r1.log") if self.replica_exchange.get() == 1 else ""))
            f.write("exit")
        self.runJob("chmod", "777 "+self._getExtraPath("launch_genesis.sh"))
        self.runJob(self._getExtraPath("launch_genesis.sh"), "")

    def createInputStep(self):

        fnVolume =self.inputVolume.get().getFileName()
        os.system("cp %s %s" %(self.inputGenesisMin.get()._getExtraPath("min.rst"), self._getExtraPath("")))
        os.system("cp %s %s" %(self.inputGenesisMin.get().inputPSF.get().getFileName(), self._getExtraPath("")))
        os.system("cp %s %s" %(self.inputGenesisMin.get().inputPDB.get().getFileName(), self._getExtraPath("")))

        pre, ext = os.path.splitext(os.path.basename(fnVolume))
        if ext != ".mrc":
            fnMRC = self._getExtraPath(pre + ".mrc")
            args = "-i " + fnVolume
            args += " --oext mrc"
            args += " -o " + fnMRC
            self.runJob("xmipp_image_convert", args)

            with mrcfile.open(fnMRC) as mrc:
                mrc_data = mrc.data
            with mrcfile.new(fnMRC, overwrite=True) as mrc:
                mrc.set_data(mrc_data)
                mrc.voxel_size = self.voxel_size.get()
                origin = -self.voxel_size.get() * np.array(mrc_data.shape) / 2
                mrc.header['origin']['x'] = origin[0]
                mrc.header['origin']['y'] = origin[1]
                mrc.header['origin']['z'] = origin[2]
                mrc.update_header_from_data()
                mrc.update_header_stats()
        else:
            fnMRC = fnVolume

        from src.density import Volume
        from src.molecule import Molecule
        m1 = Volume.from_file(file=fnMRC, sigma=2.0, cutoff=6.0)
        mol = Molecule(self.inputGenesisMin.get().inputPDB.get().getFileName())
        mol.center()
        m2 = Volume.from_coords(mol.coords, size= m1.size, voxel_size=m1.voxel_size,sigma=2.0, cutoff=6.0)

        m1.rescale(method="match", density=m2)
        if self.centerOrigin.get() ==0:
            m1.save_mrc(file=self._getExtraPath("target.mrc"), origin=None)
        else:
            m1.save_mrc(file=self._getExtraPath("target.mrc"), origin=0.0)

        prog = self.situs_dir.get() + "/bin/map2map"
        args = self._getExtraPath("target.mrc") +" "+self._getExtraPath("target.sit") +" <<< \'1\'"
        with open(self._getExtraPath("runconvert.sh"), "w") as f:
            f.write("#!/bin/bash \n")
            f.write(prog+ " "+ args+"\n")
            f.write("exit")
        self.runJob("/bin/bash", self._getExtraPath("runconvert.sh"))
        self.inputVolumeFn = self._getExtraPath("target.sit")


    def createOutputStep(self):
        if self.replica_exchange.get() ==0:
            n_outputs = self.nreplica.get()
        else:
            n_outputs=1
        pdbset = self._createSetOfPDBs("outputPDBs")
        for i in range(n_outputs):
            outputPrefix = self._getExtraPath("run_r%i" % (i+1))
            pdbset.append(AtomStruct(outputPrefix+".pdb"))

            rmsd = self.compute_rmsd_from_dcd(outputPrefix)
            cc = self.read_cc_in_log_file(outputPrefix)

            if self.molprobity.get() == 0:
                self.run_molprobity(outputPrefix)

            np.save(file=outputPrefix +"_rmsd.npy", arr=rmsd)
            np.save(file=outputPrefix +"_cc.npy", arr=cc)

        self._defineOutputs(outputPDBs=pdbset)

    def compute_rmsd_from_dcd(self, outputPrefix):
        with open(self._getExtraPath("dcd2pdb.tcl"), "w") as f:
            s=""
            s += "mol load pdb " + outputPrefix+".pdb dcd " +outputPrefix+".dcd\n"
            s += "set nf [molinfo top get numframes]\n"
            s += "for {set i 0 } {$i < $nf} {incr i} {\n"
            s += "[atomselect top all frame $i] writepdb "+outputPrefix + "tmp$i.pdb\n"
            s += "}\n"
            s += "exit\n"
            f.write(s)
        self.runJob("vmd", "-dispdev text -e "+self._getExtraPath("dcd2pdb.tcl"))

        from src.molecule import Molecule
        from src.functions import get_mol_conv, get_RMSD_coords
        rmsd = []
        target = Molecule(self.target_pdb.get().getFileName())
        N = (self.n_steps.get() // self.crdout_period.get())
        mol = Molecule(self.inputGenesisMin.get().inputPDB.get().getFileName())
        idx = get_mol_conv(mol, target, ca_only=True)
        if len(idx) > 0:
            rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
            for i in range(N):
                print(i)
                mol = Molecule(outputPrefix +"tmp"+ str(i + 1) + ".pdb")
                rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
        else:
            rmsd = np.zeros(N+1)
        # os.system("rm -f %stmp*" %(outputPrefix))
        return np.array(rmsd)

    def read_cc_in_log_file(self,outputPrefix):
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
        return np.array(cc)


    def run_molprobity(self, outputPrefix):
        os.system("~/MolProbity/cmdline/oneline-analysis %s.pdb > %s_molprobity.txt" % (outputPrefix,outputPrefix))
        with open("%s_molprobity.txt"% outputPrefix, "r") as f:
            header = None
            molprob = {}
            for i in f:
                split_line = (i.split(":"))
                if header is None:
                    if split_line[0] == "#pdbFileName":
                        header = split_line
                else:
                    if len(split_line) == len(header):
                        for i in range(len(header)):
                            molprob[header[i]] = split_line[i]

        np.savetxt(fname = "%s_molprobity.txt"% outputPrefix, X =
                np.array([float(molprob["clashscore"]),
                          float(molprob["MolProbityScore"]),
                          float(molprob["ramaFavored"]) / float(molprob["numRama"]) ,
                          float(molprob["rotaFavored"]) / float(molprob["numRota"])
                          ]))


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
