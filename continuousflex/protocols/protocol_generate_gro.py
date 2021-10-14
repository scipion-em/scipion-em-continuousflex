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


class FlexProtGenerateGRO(ProtAnalysis3D):
    """ Protocol to generate gromacs file. """
    _label = 'generate GRO'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ########################### Input ############################################
        form.addSection(label='Input')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB", important=True,
                      help='Select the PDB from which the GRO file will be generate')
        form.addParam('smogdir', params.FileParam, label="SMOG install path",
                      help='SMOG installation root directory')

        # --------------------------- INSERT steps functions --------------------------------------------


    def _insertAllSteps(self):
        self._insertFunctionStep("generateGROStep")
        self._insertFunctionStep("createOutputStep")

    def generateGROStep(self):
        import sys
        sys.path.append("/home/guest/PycharmProjects/bayesian-md-nma/")
        from src.molecule import Molecule

        inputPDB = self._getExtraPath("inputPDB.pdb")
        tmpPDB = self._getExtraPath("tmp.pdb")

        print("///////////////////////////////:")
        print("Grep ATOM ... ")
        cmd = "grep \"^ATOM\" %s > %s"% (self.inputPDB.get().getFileName(), tmpPDB)
        print(cmd)
        os.system(cmd)

        print("END ... ")
        os.system("echo \"END\" >> %s "%tmpPDB)

        print("Copying ... ")
        os.system("cp %s %s"%(tmpPDB, inputPDB))

        mol = Molecule(tmpPDB)

        print("SMOG2 ...")
        args = "-AA -i %s -dname %s " %(inputPDB,self._getExtraPath("output"))
        # args += "-limitbondlength "
        smog = self.smogdir.get() + "bin/smog2"
        self.runJob(smog, args)

    def createOutputStep(self):
        gro = EMFile(self._getExtraPath('output.gro'))
        self._defineOutputs(outputGRO=gro)
        top = EMFile(self._getExtraPath('output.top'))
        self._defineOutputs(outputTOP=top)

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