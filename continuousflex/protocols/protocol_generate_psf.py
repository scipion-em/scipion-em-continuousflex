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


class FlexProtGeneratePSF(ProtAnalysis3D):
    """ Protocol to generate PSF file. """
    _label = 'generate PSF'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ########################### Input ############################################
        form.addSection(label='Input')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB", important=True,
                      help='Select the PDB from which the PSF file will be generate')
        form.addParam('inputTopology', params.FileParam, label="Input Topology file", important=True,
                      help='CHARMM Topology file (.rtf). Can be founded at '+
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')


        # --------------------------- INSERT steps functions --------------------------------------------


    def _insertAllSteps(self):
        self._insertFunctionStep("generatePSF")
        self._insertFunctionStep("createOutputStep")


    def generatePSF(self):
        fnInputPDB = self.inputPDB.get().getFileName()
        fnPSFgen = self._getExtraPath("psfgen.tcl")
        fnOutputPDB = self._getExtraPath("output.pdb")
        fnOutputPSF = self._getExtraPath("output.psf")
        fnTopology = self.inputTopology.get()

        pre, _ = os.path.splitext(fnOutputPDB)
        with open(fnPSFgen, "w") as psfgen:
            psfgen.write("mol load pdb " + fnInputPDB + "\n")
            psfgen.write("set protein [atomselect top protein]\n")
            psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")

            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set sel [atomselect top \"pfrag $chain\"]\n")
            psfgen.write("    $sel writepdb " + pre + "_tmp${chain}.pdb\n")
            psfgen.write("}\n")

            psfgen.write("package require psfgen\n")
            psfgen.write("topology " + fnTopology + "\n")
            psfgen.write("pdbalias residue HIS HSE\n")
            psfgen.write("pdbalias residue MSE MET\n")
            psfgen.write("pdbalias atom ILE CD1 CD\n")

            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    segment U${chain} {pdb " + pre + "_tmp${chain}.pdb}\n")
            psfgen.write("    coordpdb " + pre + "_tmp${chain}.pdb U${chain}\n")
            psfgen.write("    rm -f " + pre + "_tmp${chain}.pdb\n")
            psfgen.write("}\n")

            psfgen.write("guesscoord\n")
            psfgen.write("writepdb " + fnOutputPDB+"\n")
            psfgen.write("writepsf " + fnOutputPSF+"\n")
            psfgen.write("exit\n")

        self.runJob("vmd", "-dispdev text -e "+fnPSFgen)

    def createOutputStep(self):
        pdb = AtomStruct(self._getExtraPath('output.pdb'), pseudoatoms=False)
        self._defineOutputs(outputPDB=pdb)
        psf = EMFile(self._getExtraPath('output.psf'))
        self._defineOutputs(outputPSF=psf)

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