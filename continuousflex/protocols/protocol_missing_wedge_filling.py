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
from os.path import basename, isfile
from sh_alignment.tompy.transform import fft, ifft, fftshift, ifftshift
from .utilities.spider_files3 import save_volume, open_volume
from pyworkflow.utils import replaceBaseExt
import numpy as np
from continuousflex.protocols.utilities.mwr import mwr
from continuousflex.protocols.protocol_subtomogrmas_synthesize import FlexProtSynthesizeSubtomo

REFERENCE_EXT = 0
REFERENCE_STA = 1
REFERENCE_STS = 2

METHOD_STAFILL = 0
METHOD_MCSFILL = 1


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
        group = form.addGroup('Missing-wedge parameters')
        group.addParam('tiltLow', params.IntParam, default=-60,
                       label='Lower tilt value',
                       help='The lower tilt angle used in obtaining the tilt series')
        group.addParam('tiltHigh', params.IntParam, default=60,
                       label='Upper tilt value',
                       help='The upper tilt angle used in obtaining the tilt series')
        form.addSection('Method')
        form.addParam('Method', params.EnumParam,
                      choices=['MW fill with the average', 'MW restoration using monte carlo simulation'],
                      default=METHOD_STAFILL,
                      label='Missing wedge (MW) correction method', display=params.EnumParam.DISPLAY_COMBO,
                      help='Fill the wedge by the average will use the subtomogram averaging process to fill the wedge'
                           ' of each subtomogram by the corresponding average region in Fourier space.'
                           ' The monte carlo method is an implementation of the method of E. Moebel & C. Kervrann')
        group1 = form.addGroup('MW fill with the average',
                      condition='Method==%d' % METHOD_STAFILL)
        group1.addParam('StartingReference', params.EnumParam, allowsNull=True,
                      choices=['Import an external volume', 'Select volume'],
                      default=REFERENCE_EXT,
                      label='Reference volume', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external volume file or an output volume from STA protocol')
        group1.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="File path",
                      help='Choose a reference, typically from a STA previous run')
        group1.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Volume (STA)",
                      help='Choose a reference, typically from a STA previous run')
        group1.addParam('AlignmentParameters', params.EnumParam, allowsNull=True,
                      choices=['from input metadata file', 'from STA run', 'from Subtomograms synthesize run'],
                      default=REFERENCE_EXT,
                      label='Alignment parameters', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external metadata file containing alignment parameters or STA run')
        group1.addParam('MetaDataFile', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_EXT,
                      label="Alignment parameters MetaData",
                      help='Alignment parameters, typically from a STA previous run')
        group1.addParam('MetaDataSTA', params.PointerParam,
                      pointerClass='FlexProtSubtomogramAveraging', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_STA,
                      label="Subtomogram averaging run",
                      help='Alignment parameters, typically from a STA previous run')
        group1.addParam('MetaDataSTS', params.PointerParam,
                      pointerClass='FlexProtSynthesizeSubtomo', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_STS,
                      label="Subtomogram synthesize run",
                      help='Point to the corresponding synthesize run where you imported the volumes')
        group1.addParam('applyParams', params.BooleanParam, allowsNull=True,
                      default=True,
                      label='Apply alignment after filling the wedge?',
                      help='Both aligned and none aligned versions will be kept')
        group2 = form.addGroup('MW restoration using monte carlo simulation',
                      condition='Method==%d' % METHOD_MCSFILL)
        group2.addParam('sigma_noise', params.FloatParam, default=0.2, allowsNull=True,
                       label='noise sigma',
                       help='estimated standard deviation of data noise '
                            'defines the strength of the processing (high value gives smooth images)')
        group2.addParam('T', params.IntParam, default=300, allowsNull=True,
                       label='number of iterations',
                       help='number of iterations (default: 300)')
        group2.addParam('Tb', params.IntParam, default=100, allowsNull=True,
                       label='length of the burn-in phase (Tb)',
                       help='First Tb samples are discarded (default: 100)')
        group2.addParam('beta', params.FloatParam, default=0.00004, allowsNull=True,
                       label='scale parameter (beta)',
                       help='scale parameter, affects the acceptance rate (default: 0.00004)')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/mw_filled')
        dynamoflag=False
        if self.MetaDataFile.get()==REFERENCE_STA:
            dynamoflag= self.MetaDataSTA.get().dynamoTable.get()
            print(dynamoflag)
        self._insertFunctionStep('convertInputStep')
        if self.Method == METHOD_STAFILL:
            if dynamoflag:
                self._insertFunctionStep('doAlignmentStep_STAFILL_dynamo')
                if self.applyParams.get():
                    self._insertFunctionStep('applyAlignment_dynamo')
            else:
                self._insertFunctionStep('doAlignmentStep_STAFILL')
                if self.applyParams.get():
                    self._insertFunctionStep('applyAlignment')
            self._insertFunctionStep('createOutputStep_STAFILL')
        else:
            self._insertFunctionStep('doAlignmentStep_MCSFILL')
            self._insertFunctionStep('createOutputStep_MCSFILL')
            pass

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))

    def doAlignmentStep_STAFILL_dynamo(self):
        # TODO: fix this bug
        pass
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


        copyFile(MetaDataFile, imgFn)

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
        save_volume(np.float32(MW_mask), fnmask)
        self.runJob('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
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
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            self.runJob('xmipp_image_convert', params)
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

            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            # print('xmipp_transform_geometry',params)
            # Now the STA is aligned, add the missing wedge region to the subtomogram:
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


    def doAlignmentStep_STAFILL(self):
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
        if AlignmentParameters == REFERENCE_STS:
            MetaDataSTS = self.MetaDataSTS.get()._getExtraPath('GroundTruth.xmd')
            MetaDataFile = MetaDataSTS
        copyFile(MetaDataFile, imgFn)

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
        save_volume(np.float32(MW_mask), fnmask)
        self.runJob('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
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
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            self.runJob('xmipp_image_convert', params)
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
            if AlignmentParameters != REFERENCE_STS:
                params += '--inverse'
            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
            if AlignmentParameters != REFERENCE_STS:
                params = '-i ' + tempdir + '/temp.vol -o ' + tempdir + '/temp.vol '
                params += '--rotate_volume euler 0 -90 0 '
                # print('xmipp_transform_geometry',params)
                self.runJob('xmipp_transform_geometry', params)
            # Now the STA is aligned, add the missing wedge region to the subtomogram:
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

            # for debugging, save everything that was aligned in the first iteration
            # if objId == 1:
            #     v_ave = open_volume(tempdir + '/temp.vol')
            #     save_volume(v_ave, self._getExtraPath('aligned_average_with_first_volume.spi'))

        mdImgs.write(self.imgsFn)

    def doAlignmentStep_MCSFILL(self):
        # get a copy of the input metadata
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
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
        save_volume(np.float32(MW_mask), fnmask)
        self.runJob('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
        # done creating the missing wedge mask, getting the paremeters from the form:
        sigma_noise = self.sigma_noise.get()
        T = self.T.get()
        Tb = self.Tb.get()
        beta = self.beta.get()
        # looping on all images and performing mwr
        mdImgs = md.MetaData(imgFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/mw_filled/'
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            temp_path = self._getTmpPath('temp.spi')
            # params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            params = '-i ' + imgPath + ' -o ' + temp_path + ' --type vol'
            self.runJob('xmipp_image_convert', params)
            # perform the mwr:
            # in case the file exists (continuing or injecting)
            if (isfile(new_imgPath)):
                continue
            else:
                mwr(temp_path,fnmask,new_imgPath,sigma_noise,T,Tb,beta,True)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
        mdImgs.write(self.imgsFn)

    def applyAlignment(self):
        makePath(self._getExtraPath() + '/mw_filled_aligned')
        tempdir = self._getTmpPath()
        mdImgs = md.MetaData(self.imgsFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath() + '/mw_filled_aligned/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            # rotate 90 around y, then align
            if self.AlignmentParameters.get() != REFERENCE_STS:
                params += '--rotate_volume euler 0 90 0 '
            else:
                params += '--rotate_volume euler 0 0 0 '
            self.runJob('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            if self.AlignmentParameters.get() == REFERENCE_STS:
                params += '--inverse'
            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned.xmd')
        mdImgs.write(self.fnaligned)

    def applyAlignment_dynamo(self):
        # pass
        makePath(self._getExtraPath() + '/mw_filled_aligned')
        tempdir = self._getTmpPath()
        mdImgs = md.MetaData(self.imgsFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath() + '/mw_filled_aligned/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))

            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            params += '--inverse'

            # print('xmipp_transform_geometry',params)
            self.runJob('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned.xmd')
        mdImgs.write(self.fnaligned)

    def createOutputStep_STAFILL(self):
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


    def createOutputStep_MCSFILL(self):
        partSet = self._createSetOfVolumes('not_aligned')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(MWRvolumes=partSet)


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
