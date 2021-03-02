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
from sh_alignment.tompy.transform import fft, ifft, fftshift, ifftshift
from .utilities.spider_files3 import save_volume, open_volume
from pyworkflow.utils import replaceBaseExt
import numpy as np

REFERENCE_EXT = 0
REFERENCE_STA = 1


class FlexProtMissingWedgeFilling(ProtAnalysis3D):
    """ Protocol for subtomogram missingwedge filling. """
    _label = 'missing wedge filling'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        form.addParam('StartingReference', params.EnumParam,
                      choices=['from input file', 'from STA run'],
                      default=REFERENCE_EXT,
                      label='Starting reference', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external volume file or an output volume from STA protocol')
        form.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Starting Reference Volume",
                      help='Choose a reference, typically from a STA previous run')
        form.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Subtomogram average",
                      help='Choose a reference, typically from a STA previous run')
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
        form.addParam('applyParams', params.BooleanParam,
                      default=True,
                      label='Apply alignment after filling the wedge?',
                      help='Both aligned and none aligned versions will be kept')
        form.addSection(label='Missing-wedge parameters')
        form.addParam('tiltLow', params.IntParam, default=-60,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        form.addParam('tiltHigh', params.IntParam, default=60,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath()+'/mw_filled')

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('doAlignmentStep')
        if self.applyParams.get():
            self._insertFunctionStep('applyAlignment')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))

    def doAlignmentStep(self):
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        StartingReference = self.StartingReference.get()
        ReferenceVolume = self.ReferenceVolume.get()

        if StartingReference == REFERENCE_STA:
            STAVolume = self.STAVolume.get().getFileName()
        else:
            STAVolume = ReferenceVolume

        AlignmentParameters = self.AlignmentParameters.get()
        MetaDataFile = self.MetaDataFile.get()
        if AlignmentParameters == REFERENCE_STA:
            MetaDataSTA = self.MetaDataSTA.get()._getExtraPath('final_md.xmd')
            MetaDataFile = MetaDataSTA
        copyFile(MetaDataFile,imgFn)

        tiltLow = self.tiltLow.get()
        tiltHigh = self.tiltHigh.get()

        # creating a missing-wedge mask:
        start_ang = tiltLow
        end_ang = tiltHigh
        size = self.inputVolumes.get().getDim()
        MW_mask = np.ones(size)
        x, z = np.mgrid[0.:size[0], 0.:size[2]]
        x -= size[0] / 2
        ind = np.where(x)
        z -= size[2] / 2

        angles = np.zeros(z.shape)
        angles[ind] = np.arctan(z[ind] / x[ind]) * 180 / np.pi

        angles = np.reshape(angles, (size[0], 1, size[2]))
        angles = np.repeat(angles, size[1], axis=1)

        MW_mask[angles > -start_ang] = 0
        MW_mask[angles < -end_ang] = 0

        MW_mask[size[0] // 2, :, :] = 0
        MW_mask[size[0] // 2, :, size[2] // 2] = 1
        fnmask = self._getExtraPath('Mask.spi')
        save_volume(np.float32(MW_mask),fnmask)
        self.runJob('xmipp_transform_geometry','-i '+fnmask+' --rotate_volume euler 0 90 0')
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

        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/mw_filled/'
            if index: #case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            self.runJob('xmipp_image_convert',params)
            # print('xmipp_image_convert',params)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            # Align the reference with the subtomogram:
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # print(imgPath,rot,tilt,psi,shiftx,shifty,shiftz)
            params = '-i ' + STAVolume + ' -o ' + tempdir + '/temp.vol '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            params += '--inverse'
            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + tempdir + '/temp.vol '
            params += '--rotate_volume euler 0 -90 0 '
            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            #Now the STA is aligned, add the missing wedge region to the subtomogram:
            v = open_volume(new_imgPath)
            I = fft(v)
            I = fftshift(I)
            v_ave = open_volume(tempdir + '/temp.vol')
            Iave = fft(v_ave)
            Iave = fftshift((Iave))
            Mask = open_volume(fnmask)
            Mask = np.ones(np.shape(Mask)) - Mask
            Iave = Iave * Mask
            #
            I = I + Iave
            #
            I = ifftshift(I)
            v_result = np.float32(ifft(I))
            #
            save_volume(v_result, new_imgPath)

        mdImgs.write(self.imgsFn)

    def applyAlignment(self):
        makePath(self._getExtraPath()+'/mw_filled_aligned')
        tempdir = self._getTmpPath()
        mdImgs = md.MetaData(self.imgsFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath()+'/mw_filled_aligned/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # rotate 90 around y, align, then rotate -90 to get to neutral
            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            params += '--rotate_volume euler 0 90 0 '
            self.runJob('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            # params = '-i ' + new_imgPath + ' --rotate_volume euler 0 90 0 '
            # self.runJob('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned.xmd')
        mdImgs.write(self.fnaligned)


    def createOutputStep(self):
        partSet = self._createSetOfVolumes('not_aligned')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(MissingWedgeFilledNotAligned=partSet)
        if self.applyParams.get():
            partSet2 = self._createSetOfVolumes('aligned')
            xmipp3.convert.readSetOfVolumes(self.fnaligned, partSet2)
            partSet2.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            self._defineOutputs(MissingWedgeFilledAndAligned=partSet2)
        # self._defineTransformRelation(self.inputVolumes, partSet)



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

