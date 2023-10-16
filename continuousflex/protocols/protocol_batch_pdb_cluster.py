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
from pyworkflow.protocol.params import PointerParam, FileParam, USE_GPU, GPU_LIST, BooleanParam, StringParam,LEVEL_ADVANCED
from pwem.protocols import BatchProtocol
from pwem.objects import SetOfClasses2D
from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, readSetOfVolumes
from pyworkflow.utils import runCommand
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md
import os

class FlexBatchProtClusterSet(BatchProtocol):
    """ Protocol executed when a set of cluster is created
    from set of pdbs.
    """
    _label = 'cluster set'

    def _defineParams(self, form):
        form.addHidden('inputSet', PointerParam, pointerClass='SetOfParticles,SetOfVolumes')
        form.addHidden('inputClasses', PointerParam, pointerClass='SetOfClasses2D,SetOfClasses3D')
        form.addHidden('inputPDBs', PointerParam, pointerClass='SetOfAtomStructs')
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('reconstructStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self):
        pass

    def reconstructStep(self):
        inputClasses = self.inputClasses.get()

        for i in inputClasses:
            if i.getObjId() != 0:
                classFile = self._getExtraPath("class%s.xmd" % str(i.getObjId()).zfill(6))
                if isinstance(inputClasses, SetOfClasses2D):
                    writeSetOfParticles(i, classFile)
                else:
                    writeSetOfVolumes(i,classFile)

        for i in inputClasses:
            if i.getObjId() != 0:
                classFile = self._getExtraPath("class%s.xmd" % str(i.getObjId()).zfill(6))
                classVol = self._getExtraPath("class%s.vol" % str(i.getObjId()).zfill(6))
                if isinstance(inputClasses, SetOfClasses2D):
                    args = "-i %s -o %s " % (classFile, classVol)
                    args += ' --sampling %f' % self.inputClasses.get().getSamplingRate()

                    if self.useGpu.get():
                        args += ' --thr %d' % self.numberOfThreads.get()

                    if self.useGpu.get():
                        if self.numberOfMpi.get() > 1:
                            self.runJob('xmipp_cuda_reconstruct_fourier', args,
                                        numberOfMpi=len((self.gpuList.get()).split(',')) + 1)
                        else:
                            self.runJob('xmipp_cuda_reconstruct_fourier', args)
                    else:
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
        inputClasses = self.inputClasses.get()
        for i in inputClasses:
            if i.getObjId() != 0:
                classVol = self._getExtraPath("class%s.vol" % str(i.getObjId()).zfill(6))
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
