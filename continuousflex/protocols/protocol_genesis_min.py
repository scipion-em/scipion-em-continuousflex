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


class FlexProtGenesisMin(ProtAnalysis3D):
    """ Protocol for minimizing a PDB using Genesis. """
    _label = 'genesis min'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB", important=True,
                      help='Select the input PDB.')
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

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("minimizeStep")
        self._insertFunctionStep("generateOutputPDBStep")
        self._insertFunctionStep("createOutputStep")

    def minimizeStep(self):
        s = ""
        s += "\n[INPUT]\n"
        s += "topfile = " + self.inputRTF.get() + "\n"
        s += "parfile = " + self.inputPRM.get() + "\n"
        s += "pdbfile = " + self.inputPDB.get().getFileName() + "\n"
        s += "psffile = " + self.inputPSF.get().getFileName() + "\n"
        s += "\n[OUTPUT]\n"
        s += "dcdfile = " + self._getExtraPath("min.dcd") + "\n"
        s += "rstfile = " + self._getExtraPath("min.rst") + "\n"
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

        with open(self._getExtraPath("min"), "w") as f:
            f.write(s)

        self.runJob(self.genesisDir.get()+"/bin/atdyn", self._getExtraPath("min"))

    def generateOutputPDBStep(self):
        with open(self._getExtraPath("dcd2pdb.tcl"), "w") as f:
            s=""
            s += "mol load pdb " + self.inputPDB.get().getFileName() + " dcd " +self._getExtraPath("min.dcd")+ "\n"
            s += "set nf [molinfo top get numframes]\n"
            s += "for {set i 0 } {$i < $nf} {incr i} {\n"
            s += "[atomselect top all frame $i] writepdb "+self._getExtraPath("output.pdb")+"\n"
            s += "}\n"
            s += "exit\n"
            f.write(s)
        self.runJob("vmd", "-dispdev text -e "+self._getExtraPath("dcd2pdb.tcl"))

    def createOutputStep(self):
        self._defineOutputs(outputPDB= AtomStruct(self._getExtraPath("output.pdb")))
        rst = EMFile(self._getExtraPath('min.rst'))
        self._defineOutputs(outputRST=rst)




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