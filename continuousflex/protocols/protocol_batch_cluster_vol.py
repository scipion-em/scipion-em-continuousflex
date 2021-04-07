# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
# *             Slavica Jonic                (slavica.jonic@upmc.fr)
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


from pyworkflow.protocol.params import PointerParam, FileParam, IntParam
from pwem.protocols import BatchProtocol
from pwem.objects import Volume, SetOfVolumes
from xmipp3.convert import writeSetOfVolumes
import pwem.emlib.metadata as md
import os


class FlexBatchProtNMAClusterVol(BatchProtocol):
    """ Protocol executed when a cluster is created
    from NMA volumes and theirs deformations.
    """
    _label = 'nma vol cluster'

    def _defineParams(self, form):
        form.addHidden('inputNmaDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)
        form.addHidden('angleYflag', IntParam)

    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        outputVol = self._getExtraPath('reconstruction.vol')

        self._insertFunctionStep('convertInputStep', volumesMd)
        self._insertFunctionStep('averagingStep')
        self._insertFunctionStep('createOutputStep', outputVol)

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, volumesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        inputSet = self.inputNmaDimred.get().getInputParticles()
        partSet = self._createSetOfVolumes()
        partSet.copyInfo(inputSet)
        tmpSet = SetOfVolumes(filename=self.sqliteFile.get())
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()

        self._defineOutputs(OutputVolumes=partSet)
        self._defineTransformRelation(inputSet, partSet)
        writeSetOfVolumes(partSet, volumesMd)


    def averagingStep(self):
        flag = self.angleYflag.get()

        volumesMd = self._getExtraPath('volumes.xmd')
        mdVols = md.MetaData(volumesMd)

        counter = 0
        first = True
        for objId in mdVols:
            counter = counter + 1
            imgPath = mdVols.getValue(md.MDL_IMAGE, objId)

            rot = mdVols.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdVols.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdVols.getValue(md.MDL_ANGLE_PSI, objId)
            x_shift = mdVols.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdVols.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdVols.getValue(md.MDL_SHIFT_Z, objId)

            # The flip one is used here to determine if we need to use the option --inverse
            # with xmipp_transform_geometry
            flip = mdVols.getValue(md.MDL_ANGLE_Y, objId)

            outputVol = self._getExtraPath('reconstruction.vol')
            tempVol = self._getExtraPath('temp.vol')
            extra = self._getExtraPath()

            if flag == 0 :
                if first:
                    print("THERE IS NO COMPENSATION FOR THE MISSING WEDGE")
                    first = False

                params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            else:
                if first:
                    print("THERE IS A COMPENSATION FOR THE MISSING WEDGE")
                    first = False
                # First got to rotate each volume 90 degrees about the y axis, align it, then rotate back and sum it
                params = '-i %(imgPath)s -o %(tempVol)s --rotate_volume euler 0 90 0' % locals()
                self.runJob('xmipp_transform_geometry', params)
                params = '-i %(tempVol)s -o %(tempVol)s --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s ' % locals()

            self.runJob('xmipp_transform_geometry', params)

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
        #outputParticles
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
        return []

    def _methods(self):
        return []

