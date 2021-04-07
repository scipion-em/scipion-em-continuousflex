# **************************************************************************
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
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

from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath, copyFile
from os.path import basename


REFERENCE_EXT = 0
REFERENCE_STA = 1


class FlexProtApplyVolSetAlignment(ProtAnalysis3D):
    """ Protocol for subtomogram alignment after STA """
    _label = 'apply subtomogram alignment'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        form.addParam('AlignmentParameters', params.EnumParam,
                      choices=['from input file', 'from STA run'],
                      default=REFERENCE_EXT,
                      label='Alignment parameters', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external metadata file containing alignment parameters or STA run')
        form.addParam('MetaDataFile', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_EXT,
                      label="Alignment parameters MetaData",
                      help='Alignment parameters, typically from a STA previous run')
        form.addParam('MetaDataSTA', params.PointerParam,
                      pointerClass='FlexProtSubtomogramAveraging', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_STA,
                      label="Subtomogram averaging run",
                      help='Alignment parameters, typically from a STA previous run')
        form.addParam('angleY', params.BooleanParam,
                      default=True,
                      label='Are those parameters come from Scipion/Xmipp?',
                      help='If the original alignment was done on Dynamo or if the alignment was done '
                           'without missing wedge compensation, switch this to no')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('prepareMetaData')
        self._insertFunctionStep('applyAlignment')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))

    def prepareMetaData(self):
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        AlignmentParameters = self.AlignmentParameters.get()
        MetaDataFile = self.MetaDataFile.get()
        if AlignmentParameters == REFERENCE_STA:
            MetaDataSTA = self.MetaDataSTA.get()._getExtraPath('final_md.xmd')
            MetaDataFile = MetaDataSTA
        copyFile(MetaDataFile,imgFn)

        mdImgs = md.MetaData(imgFn)
        # in case of metadata from an external file, it has to be updated with the proper filenames from 'input.xmd'
        inputSet = self.inputVolumes.get()

        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fn = xmipp3.convert.xmippToLocation(imgPath)
            if (index):  # case the input is a stack
                # Conside the index is the id in the input set
                particle = inputSet[index]
            else:  # input is not a stack
                # convert the inputSet to metadata:
                mdtemp = md.MetaData(self._getExtraPath('input.xmd'))
                # Loop and find the index based on the basename:
                bn_retrieved = basename(imgPath)
                for searched_index in mdtemp:
                    imgPath_temp = mdtemp.getValue(md.MDL_IMAGE, searched_index)
                    bn_searched = basename(imgPath_temp)
                    if bn_searched == bn_retrieved:
                        index = searched_index
                        particle = inputSet[index]
                        break
            mdImgs.setValue(md.MDL_IMAGE, xmipp3.convert.getImageLocation(particle), objId)
            mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        mdImgs.write(self.imgsFn)


    def applyAlignment(self):
        makePath(self._getExtraPath()+'/aligned')
        tempdir = self._getTmpPath()
        mdImgs = md.MetaData(self.imgsFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath()+'/aligned/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # rotate 90 around y, align, then rotate -90 to get to neutral
            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            if(self.angleY):
                params += '--rotate_volume euler 0 90 0 '
            else: # only to convert
                params += '--rotate_volume euler 0 0 0 '
            self.runJob('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            if (not(self.angleY)):
                params += ' --inverse '

            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            # params = '-i ' + new_imgPath + ' --rotate_volume euler 0 -90 0 '
            # self.runJob('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned.xmd')
        mdImgs.write(self.fnaligned)


    def createOutputStep(self):
        partSet = self._createSetOfVolumes('aligned')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes_aligned.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(outputVolumes=partSet)


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return []

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()
