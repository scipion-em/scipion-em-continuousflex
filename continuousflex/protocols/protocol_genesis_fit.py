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
        form.addSection(label='Input')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation')
        form.addParam('forcefield', params.EnumParam, label="Forcefield", default=0,
                      choices=['CHARMM', 'AAGO'],
                      help="TODo")
        form.addParam('inputGenesisMin', params.PointerParam,
                      pointerClass='FlexProtGenesisMin', label="Input Genesis Minimization",
                      help='Select the input minimization', condition="forcefield==0")
        form.addParam('inputGRO', params.PointerParam, label="Coordinate file .gro",
                      pointerClass='EMFile', help='TODO', condition="forcefield==1")
        form.addParam('inputTOP', params.PointerParam, label="Topology file .top",
                      pointerClass='EMFile', help='TODO', condition="forcefield==1")
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
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Select the number of steps in the MD fitting")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="TODO")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinates output period',
                      help="TODO")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="TODO")
        form.addParam('constantK', params.IntParam, default=10000, label='Force constant K',
                      help="TODO")
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=0,
                      choices=['Yes', 'NO'],
                      help="TODo")
        form.addParam('centerOrigin', params.EnumParam, label="Center Origin", default=0,
                      choices=['Yes', 'NO'],
                      help="TODo")
        form.addParam('time_step', params.FloatParam, default=0.001, label='Time step (ps)',
                      help="TODO")
        form.addParam('n_proc', params.IntParam, default=1, label='Number of processors',
                      help="TODO")
        form.addParam('n_rep', params.IntParam, default=1, label='[EXP] Number of repetition',
                      help="EXP")
        form.addParam('target_pdb', params.PointerParam,
                      pointerClass='AtomStruct', label="[EXP] Target PDB", help='EXP')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of normal mode vectors computed by normal mode analysis.')
        form.addParam('n_modes', params.IntParam, default=3, label='Number of modes',
                      help="TODO")
        form.addParam('first_mode', params.IntParam, default=7, label='First mode',
                      help="TODO")
        form.addParam('global_dt', params.FloatParam, default=10.0, label='Global dt',
                      help="TODO")
        form.addParam('fitGlobal', params.EnumParam, label="Fit Global ?", default=0,
                      choices=['Yes', 'No'],
                      help="TODo")
        form.addParam('tpcontrol', params.EnumParam, label="Temperature control", default=0,
                      choices=['LANGEVIN', 'BERENDSEN', 'NO'],
                      help="TODo")

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("createInputStep")
        self.niter=0
        self.times = []
        for i in range(self.n_rep.get()):
            self._insertFunctionStep("fittingStep")
            self._insertFunctionStep("createOutputStep")

    def fittingStep(self):
        min = self.inputGenesisMin.get()
        outputPrefix = self._getExtraPath("run"+str(self.niter)+"_")
        if self.tpcontrol.get() == TPCONTROL_LANGEVIN:
            tpcontrol = "LANGEVIN"
        elif self.tpcontrol.get() == TPCONTROL_BERENDSEN:
            tpcontrol = "BERENDSEN"
        else:
            tpcontrol = "NO"

        s = "[INPUT] \n"
        if self.forcefield == 0:
            s += "topfile = "+min.inputRTF.get()+"\n"
            s += "parfile = "+min.inputPRM.get()+"\n"
            s += "pdbfile = "+min.inputPDB.get().getFileName()+"\n"
            s += "psffile = "+min.inputPSF.get().getFileName()+"\n"
            s += "rstfile = "+min._getExtraPath("min.rst")+"\n"
        elif self.forcefield == 1:
            s += "grotopfile = " + self.inputTOP.get().getFileName() + "\n"
            s += "grocrdfile = " + self.inputGRO.get().getFileName() + "\n"

        s += "[OUTPUT] \n"
        s += "dcdfile = "+outputPrefix + ".dcd\n"
        s += "rstfile = "+outputPrefix + ".rst\n"
        s += "pdbfile = "+outputPrefix + ".pdb\n"

        s += "[ENERGY] \n"
        if self.forcefield == 0:
            s += "forcefield = CHARMM  # CHARMM force field\n"
        elif self.forcefield == 1:
            s += "forcefield = AAGO  # AAGO\n"
        s += "electrostatic = CUTOFF  # use cutoff scheme for non-bonded terms \n"
        s += "switchdist = 10.0  # switch distance \n"
        s += "cutoffdist = 13.0  # cutoff distance \n"
        s += "pairlistdist = 16.0  # pair-list distance \n"
        s += "vdw_force_switch = YES \n"
        if self.implicitSolvent == 0:
            s += "implicit_solvent = GBSA    # [GBSA] \n"
            s += "gbsa_eps_solvent = 78.5    # solvent dielectric constant in GB \n"
            s += "gbsa_eps_solute  = 1.0     # solute dielectric constant in GB \n"
            s += "gbsa_salt_cons   = 0.2     # salt concentration (mol/L) in GB \n"
            s += "gbsa_surf_tens   = 0.005   # surface tension (kcal/mol/A^2) in SA \n"
        else:
            s += "implicit_solvent = NONE    # [None] \n"

        s += "[DYNAMICS] \n"
        s += "integrator = VVER  # [LEAP,VVER] \n"
        s += "nsteps = "+str(self.n_steps.get())+" \n"
        s += "timestep = "+str(self.time_step.get())+"  #\n"
        s += "eneout_period = "+str(self.eneout_period.get())+" \n"
        s += "crdout_period = "+str(self.crdout_period.get())+" \n"
        s += "rstout_period = "+str(self.n_steps.get())+"\n"
        s += "nbupdate_period = "+str(self.nbupdate_period.get())+"\n"
        s += "iseed = "+str(np.random.randint(1, 31415))+"  # random number seed  \n"

        s += "[CONSTRAINTS] \n"
        s += "rigid_bond = NO  # use SHAKE \n"

        s += "[ENSEMBLE] \n"
        s += "ensemble = NVT  # constant temperature \n"
        s += "tpcontrol = "+tpcontrol+"  # Langevin thermostat \n"
        s += "temperature = 300  # T = 300 K \n"
        # s += "gamma_t = 5  # friction coefficient (ps-1) \n"

        s += "[BOUNDARY] \n"
        s += "type = NOBC  # No periodic boundary condition \n"

        s += "[SELECTION] \n"
        s += "group1 = all and not hydrogen\n"

        s += "[RESTRAINTS] \n"
        s += "nfunctions = 1 \n"
        s += "function1 = EM  # apply restraints from EM density map \n"
        s += "constant1 = "+str(self.constantK.get())+" \n"
        s += "select_index1 = 1  # apply restraint force on protein heavy atoms \n"

        s += "[EXPERIMENTS] \n"
        s += "emfit = YES  # perform EM flexible fitting \n"
        s += "emfit_target = "+self.inputVolumeFn+"\n"
        s += "emfit_sigma = 2.0  # half of the map resolution (5 A) \n"
        s += "emfit_tolerance = 0.01  # Tolerance for error (0.1%) \n"
        s += "emfit_period = 1  # emfit force update period \n"
        s += "emfit_nma = "+self._getExtraPath("emfit_nma")+"\n"

        with open(self._getExtraPath("fitting"), "w") as f:
            f.write(s)

        with open(self._getExtraPath("emfit_nma"), "w") as f:
            f.write(str(self.fitGlobal.get())+"\n")
            f.write(os.path.splitext(self.inputModes.get().getFileName())[0]+"/vec.\n")
            f.write(str(self.n_modes.get())+"\n")
            f.write(str(self.first_mode.get())+"\n")
            f.write(str(self.global_dt.get())+"\n")

        with open(self._getExtraPath("launch_genesis.sh"), "w") as f:
            f.write("export OMP_NUM_THREADS="+str(self.n_proc.get())+"\n")
            f.write("echo \"OMP NUM THREADS : \"\n")
            f.write("echo $OMP_NUM_THREADS\n")
            f.write(self.genesisDir.get()+"/bin/atdyn ")
            f.write(self._getExtraPath("fitting")+"\n")
            f.write("exit")
        t=time.time()
        self.runJob("chmod", "777 "+self._getExtraPath("launch_genesis.sh"))
        self.runJob(self._getExtraPath("launch_genesis.sh"), "")
        self.times.append(time.time()-t)

    def createInputStep(self):

        fnVolume =self.inputVolume.get().getFileName()

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
        outputPrefix = self._getExtraPath("run"+str(self.niter)+"_")
        self._defineOutputs(outputPDB= AtomStruct(outputPrefix+".pdb"))
        with open(self._getExtraPath("dcd2pdb.tcl"), "w") as f:
            s=""
            s += "mol load pdb " + outputPrefix+".pdb dcd " +outputPrefix+".dcd\n"
            s += "set nf [molinfo top get numframes]\n"
            s += "for {set i 0 } {$i < $nf} {incr i} {\n"
            s += "[atomselect top all frame $i] writepdb "+outputPrefix + "$i.pdb\n"
            s += "}\n"
            s += "exit\n"
            f.write(s)
        self.runJob("vmd", "-dispdev text -e "+self._getExtraPath("dcd2pdb.tcl"))

        self.compute_cc_rmsd()
        self.niter += 1
        np.save(file=self._getExtraPath("times.npy"), arr =np.array(self.times))

    def compute_cc_rmsd(self):
        outputPrefix = self._getExtraPath("run" + str(self.niter) + "_")
        print(self.n_steps.get())
        print(self.crdout_period.get())
        cc, rmsd = get_cc_rmsd(N=(self.n_steps.get()//self.crdout_period.get()), prefix=outputPrefix,
                               target=Molecule(self.target_pdb.get().getFileName()),
                size=100, voxel_size=2.0, cutoff=6.0, sigma=2.0, step=1, test_idx=True)
        fig, ax = plt.subplots(1,2)
        ax[0].set_xlabel("MD step")
        ax[0].set_ylabel("CC")
        ax[0].set_title("Cross correlation")
        ax[0].plot(cc)
        ax[1].set_xlabel("MD step")
        ax[1].set_ylabel("RMSD (A)")
        ax[1].set_title("Root Mean Square Deviation")
        ax[1].plot(rmsd)
        fig.savefig(outputPrefix+".png")


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
