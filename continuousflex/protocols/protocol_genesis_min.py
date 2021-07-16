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
from pwem.objects.data import AtomStruct, EMFile

import os
import numpy as np

class FlexProtGenesisMin(ProtAnalysis3D):
    """ Protocol for minimizing a PDB using Genesis. """
    _label = 'genesis min'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct, SetOfPDBs, SetOfAtomStructs', label="Input PDB", important=True,
                      help='Select the input PDB or set of PDBs.')
        form.addParam('inputPSF', params.PointerParam, label="Protein Structure File (PSF)",
                      pointerClass='EMFile', help='Structure file (.psf). Can be generated with generatePSF protocol ')
        form.addParam('inputPRM', params.FileParam, label="Parameter File (PRM)",
                      help='CHARMM force field parameter file (.prm). Can be founded at '+
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')
        form.addParam('inputRTF', params.FileParam, label="Topology File (RTF)",
                      help='CHARMM force field topology file (.rtf). Can be founded at '+
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')
        form.addParam('n_steps', params.IntParam, default=1000, label='Number of steps',
                      help="Select the number of steps in the minimization steepest descend.")
        form.addParam('molprobity', params.EnumParam, label="Do Molprobity ?", default=0,
                      choices=['Yes', 'No'],
                      help="TODO")
        form.addParam('n_proc', params.IntParam, default=1, label='Number of processors',
                      help="TODO")
        form.addParam('n_threads', params.IntParam, default=1, label='Number of threads',
                      help="TODO")

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        if isinstance(self.inputPDB.get(), AtomStruct):
            self.nPDBs = 1
            self.inputPDBfname = [self.inputPDB.get().getFileName()]
            self.outputPrefix = [self._getExtraPath("min")]
        else:
            self.nPDBs = 0
            self.inputPDBfname=[]
            self.outputPrefix=[]
            for i in self.inputPDB.get():
                self.nPDBs +=1
                self.inputPDBfname.append(i.getFileName())
                self.outputPrefix.append(self._getExtraPath("min%i"%self.nPDBs))

            print("///////////////////////////////////////////////////////////////////////////////")
            print(self.inputPDBfname)
            print(self.outputPrefix)


        self._insertFunctionStep("minimizeStep")
        self._insertFunctionStep("generateOutputPDBStep")
        self._insertFunctionStep("createOutputStep")


    def minimizeStep(self):
        for i in range(self.nPDBs):
            s = ""
            s += "\n[INPUT]\n"
            s += "topfile = " + self.inputRTF.get() + "\n"
            s += "parfile = " + self.inputPRM.get() + "\n"
            s += "pdbfile = " + self.inputPDBfname[i] + "\n"
            s += "psffile = " + self.inputPSF.get().getFileName() + "\n"
            s += "\n[OUTPUT]\n"
            s += "dcdfile = %s.dcd \n" % self.outputPrefix[i]
            s += "rstfile = %s.rst \n" % self.outputPrefix[i]
            s += "\n[ENERGY]\n"
            s += "forcefield = CHARMM  # CHARMM force field\n"
            s += "electrostatic = CUTOFF  # use cutoff scheme for non-bonded terms\n"
            s += "switchdist = 23.0  # switch distance\n"
            s += "cutoffdist = 25.0  # cutoff distance\n"
            s += "pairlistdist = 27.0  # pair-list distance\n"
            s += "implicit_solvent = GBSA  # use GBSA implicit solvent model\n"
            s += "gbsa_salt_cons = 0.15  # salt concentration\n"
            s += "gbsa_surf_tens = 0.005  # surface tension coefficient in SA term\n"
            s += "gbsa_eps_solvent = 78.5  # dielectric constant of solvent\n"
            s += "vdw_force_switch = YES\n"
            s += "\n[MINIMIZE]\n"
            s += "method = SD  # Steepest descent\n"
            s += "nsteps = "+ str(self.n_steps.get()) + "\n"
            s += "eneout_period = 10  # energy output period\n"
            s += "crdout_period = " + str(self.n_steps.get()) + "\n"
            s += "rstout_period = " + str(self.n_steps.get()) + "\n"
            s += "nbupdate_period = 10  # nonbond update period\n"
            s += "[BOUNDARY] \n"
            s += "type = NOBC  # No periodic boundary condition \n"

            with open(self.outputPrefix[i], "w") as f:
                f.write(s)

            with open(self._getExtraPath("launch_genesis.sh"), "w") as f:
                f.write("export OMP_NUM_THREADS=" + str(self.n_threads.get()) + "\n")
                f.write("echo \"OMP NUM THREADS : \"\n")
                f.write("echo $OMP_NUM_THREADS\n")
                f.write("mpirun -np %s %s/bin/atdyn %s\n" %
                        (self.n_proc.get(), self.genesisDir.get(), self.outputPrefix[i]))
                f.write("exit")
            self.runJob("chmod", "777 " + self._getExtraPath("launch_genesis.sh"))
            self.runJob(self._getExtraPath("launch_genesis.sh"), "")

    def generateOutputPDBStep(self):
        for i in range(self.nPDBs):
            with open(self._getExtraPath("dcd2pdb.tcl"), "w") as f:
                s=""
                s += "mol load pdb %s dcd %s.dcd \n" % (self.inputPDBfname[i], self.outputPrefix[i])
                s += "set nf [molinfo top get numframes]\n"
                s += "for {set i 0 } {$i < $nf} {incr i} {\n"
                s += "[atomselect top all frame $i] writepdb %s.pdb\n" % self.outputPrefix[i]
                s += "}\n"
                s += "exit\n"
                f.write(s)
            self.runJob("vmd", "-dispdev text -e "+self._getExtraPath("dcd2pdb.tcl"))

    def createOutputStep(self):
        for i in range(self.nPDBs):
            self._defineOutputs(outputPDB= AtomStruct("%s.pdb" % self.outputPrefix[i]))
            rst = EMFile("%s.rst" % self.outputPrefix[i])
            self._defineOutputs(outputRST=rst)
            if self.molprobity.get() == 0:
                self.run_molprobity(self.outputPrefix[i])

    # def createOutputSetStep(self, N):
    #     pdbset = self._createSetOfPDBs("mins")
    #     for i in range(N):

    def run_molprobity(self, outputPrefix):
        self.runJob("~/MolProbity/cmdline/oneline-analysis", "%s.pdb > %s_molprobity.txt" % (outputPrefix,outputPrefix))
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

        print("/////////////////////////////////////////////////////")
        print(molprob)

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