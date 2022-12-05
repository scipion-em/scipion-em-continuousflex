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

import multiprocessing
from pyworkflow.protocol.params import PointerParam, FileParam
from pwem.protocols import BatchProtocol
from pwem.objects import SetOfClasses2D
from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, readSetOfVolumes

from pyworkflow.utils import runCommand
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md


class FlexBatchProtClusterSet(BatchProtocol):
    """ Protocol executed when a set of cluster is created
    from set of pdbs.
    """
    _label = 'cluster set'

    def _defineParams(self, form):
        form.addHidden('inputSet', PointerParam, pointerClass='SetOfClasses2D,SetOfClasses3D')
        form.addHidden('inputSet', PointerParam, pointerClass='SetOfClasses2D,SetOfClasses3D')
        form.addParallelSection(threads=1, mpi=multiprocessing.cpu_count()//2-1)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('reconstructStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self):
        pass

    def reconstructStep(self):
        inputClasses = self.inputSet.get()

        for i in inputClasses:
            if i.getObjId() != 0:
                classFile = self._getExtraPath("class%i.xmd" % i.getObjId())
                if isinstance(inputClasses, SetOfClasses2D):
                    writeSetOfParticles(i, classFile)
                else:
                    writeSetOfVolumes(i,classFile)

        for i in inputClasses:
            if i.getObjId() != 0:
                classFile = self._getExtraPath("class%i.xmd" % i.getObjId())
                classVol = self._getExtraPath("class%i.vol" % i.getObjId())
                if isinstance(inputClasses, SetOfClasses2D):
                    args = "-i %s -o %s " % (classFile, classVol)
                    if self.numberOfMpi.get() > 1 :
                        progname = "xmipp_mpi_reconstruct_fourier "
                        self.runJob(progname, args)
                    else:
                        progname = "xmipp_reconstruct_fourier "
                        runCommand(progname + args)
                else:
                    classAvg = ImageHandler().computeAverage(i)
                    classAvg.write(classVol)

    def createOutputStep(self):
        outputMd = md.MetaData()
        inputClasses = self.inputSet.get()
        for i in inputClasses:
            if i.getObjId() != 0:
                classVol = self._getExtraPath("class%i.vol" % i.getObjId())
                index = outputMd.addObject()
                outputMd.setValue(md.MDL_IMAGE, classVol, index)
                outputMd.setValue(md.MDL_ITEM_ID, i.getObjId(), index)
        outputMd.write(self._getExtraPath("outputVols.xmd"))
        outputVols = self._createSetOfVolumes()
        readSetOfVolumes(self._getExtraPath("outputVols.xmd"),outputVols)
        outputVols.setSamplingRate(inputClasses.getSamplingRate())
        self._defineOutputs(outputVols=outputVols)
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022continuousflex']

    def _methods(self):
        return []
