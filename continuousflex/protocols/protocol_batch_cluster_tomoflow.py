# **************************************************************************
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
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
# *
# **************************************************************************

from pyworkflow.protocol.params import PointerParam, FileParam
from pwem.protocols import BatchProtocol
from pwem.objects import Volume, SetOfVolumes
from xmipp3.convert import writeSetOfVolumes
import pwem.emlib.metadata as md
import os


class FlexBatchProtTomoFlowCluster(BatchProtocol):
    """ Protocol executed when a cluster is created
    from TomoFlow dimred.
    """
    _label = 'tomoflow vol cluster'

    def _defineParams(self, form):
        form.addHidden('inputHeteroFlowDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)

    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        outputVol = self._getExtraPath('average.spi')

        self._insertFunctionStep('convertInputStep', volumesMd)
        self._insertFunctionStep('averagingStep')
        self._insertFunctionStep('createOutputStep', outputVol)

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, volumesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        # inputSet = self.inputHeteroFlowDimred.get().getInputParticles().get()
        inputSet = self.inputHeteroFlowDimred.get().getInputParticles()
        partSet = self._createSetOfVolumes()
        # partSet = SetOfVolumes()
        partSet.copyInfo(inputSet)
        tmpSet = SetOfVolumes(filename=self.sqliteFile.get())
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()

        self._defineOutputs(OutputVolumes=partSet)
        # self._defineTransformRelation(inputSet, partSet)
        writeSetOfVolumes(partSet, volumesMd)


    def averagingStep(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        mdVols = md.MetaData(volumesMd)
        counter = 0
        for objId in mdVols:
            counter = counter + 1
            imgPath = mdVols.getValue(md.MDL_IMAGE, objId)
            outputVol = self._getExtraPath('average.spi')
            tempVol = self._getExtraPath('temp.spi')
            extra = self._getExtraPath()

            params = '-i %(imgPath)s -o %(tempVol)s --type vol ' % locals()
            self.runJob('xmipp_image_convert',params)

            if counter == 1 :
                os.system("mv %(tempVol)s %(outputVol)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(outputVol)s -o %(outputVol)s ' % locals()
                self.runJob('xmipp_image_operate', params)

        params = '-i %(outputVol)s --divide %(counter)s -o %(outputVol)s ' % locals()
        self.runJob('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())


    def createOutputStep(self, outputVol):
        vol = Volume()
        vol.setFileName(outputVol)
        vol.setSamplingRate(self.OutputVolumes.getSamplingRate())
        self._defineOutputs(outputVol=vol)

    #--------------------------- INFO functions --------------------------------------------
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

