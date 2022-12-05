# **************************************************************************
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
# *
# * IMPMC Sorbonne University
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

from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath, copyFile, cleanPath
from sh_alignment.tompy.transform import fft, ifft, fftshift, ifftshift
from pyworkflow.utils import replaceBaseExt
from .utilities.spider_files3 import *
import os
from os.path import basename, isfile
from pwem.utils import runProgram
from pwem import Domain
from pwem.objects import Volume
from joblib import Parallel, delayed
import continuousflex
from subprocess import check_call
from pwem.emlib.image import ImageHandler
from .convert import eulerAngles2matrix, matrix2eulerAngles
from pyworkflow.utils import getListFromRangeString
import multiprocessing
from continuousflex import Plugin
import joblib

REFERENCE_EXT = 0
REFERENCE_STA = 1


class FlexProtRefineSubtomoAlign(ProtAnalysis3D):
    """ Protocol for refining subtomogram alignment and filling the missing wedge based on optical flow and Fast Rotational Matching (FRM).
     The protocol takes as input a set of subtomograms, with their subtomogram averaging protocol.
     It uses this global subtomogrm average to fill the missing wedge in Fourier space (the missing wedge is replaced by the corresponding region from the global average).
     Optical flow is used to match the global average with each of the missing wedge filled and aligned subtomograms (matched subtomograms are generated).
     Rigid-body alignment is performed using FRM from the matched subtomogram, and the rigid-body alignment for the input subtomograms is updated.
     Few iterations are usually sufficient (1-5), and the rigid-body alignment will be refined"""
    _label = 'refine subtomogram alignment'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Settings')
        group = form.addGroup('Choose what processes you want to perform:')
        group.addParam('FillWedge', params.BooleanParam, default=True,
                       label='Correct the missing wedge?',
                       help='This step will correct the missing wedge by filling it with the average'
                            ' of each iteration.')
        group.addParam('tiltLow', params.IntParam, default=-60,
                      condition='FillWedge==%d' % True,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        group.addParam('tiltHigh', params.IntParam, default=60,
                      condition='FillWedge==%d' % True,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')
        group.addParam('Alignment_refine', params.BooleanParam, default=True,
                       label='Refine the rigid-body alignment?',
                       help='This step will refine the rigid-body alignment given a previous run of subtomogram averaging'
                            ' in the workspace')
        group.addParam('NumOfIters', params.IntParam, default=3,
                       condition='Alignment_refine',
                       label='Refinment iterations', help='How many times you want to iterate to perform'
                                                         ' subtomogram alignment refinement.')
        group.addParam('KeepFiles', params.BooleanParam, default=False,
                       expertLevel=params.LEVEL_ADVANCED,
                       label='Keep the intermediate files on the disk (CAREFUL!)?',
                       condition='Alignment_refine',
                       help='This will keep all the itermediate files on the disk (useful for debugging). Be very careful'
                            ' that it requires 6 times the size of the input subtomograms per iteration.'
                            ' For example, if you are using 1GB size input subtomogams, and you are refining for 4 iterations,'
                            ' then this requires 24GB on the disk; where setting this to no, we require 6GB.')
        group.addParam('ApplyAlignment', params.EnumParam,
                       label='Apply volume/subtomogam alignment?',
                       choices=['Yes'],
                       default=REFERENCE_EXT,
                       display=params.EnumParam.DISPLAY_HLIST,
                       help='This protocol by default applies previous StA alignment on the subtomograms.')

        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      label="Input volumes/subtomograms", important=True,
                      help='Select volumes')
        group = form.addGroup('Reference volume (last iteration average of StA) and a Mask',
                              condition='Alignment_refine or FillWedge')
        group.addParam('StartingReference', params.EnumParam,
                      choices=['Browse for an external volume file', 'Select a volume from the project workspace'],
                      default=REFERENCE_STA,
                      label='Average volume', display=params.EnumParam.DISPLAY_COMBO)
        group.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Volume path",
                      help='Choose a reference, typically from a StA previous run')
        group.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Selected volume",
                      help='Choose a reference, typically from a StA previous run')
        group.addParam('applyMask', params.BooleanParam, label='Use a mask?', default=False,
                       help='A mask that can be applied on the reference without cropping it. The same mask will be'
                            ' applied on the aligned subtomograms at each iteration (do not apply this mask in advance)'
                       )
        group.addParam('Mask', params.PointerParam,
                       condition='applyMask',
                       pointerClass='Volume', allowsNull=True,
                       label="Select mask")
        group = form.addGroup('Alignment parameters: last iteration table of StA (Scipion/Xmipp metadata)')
        group.addParam('AlignmentParameters', params.EnumParam,
                      choices=['Browse for a file', 'Select a subtomogram averaging protocol '
                                                    'from the project workspace'],
                      default=REFERENCE_STA,
                      label='Alignment parameters', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external metadata file containing alignment parameters or StA run')
        group.addParam('MetaDataFile', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_EXT,
                      label="File for rigid-body alignment parameters (Xmipp/Scipion MetaData)",
                      help='Alignment parameters, typically from a StA previous run')
        group.addParam('MetaDataSTA', params.PointerParam,
                      pointerClass='FlexProtSubtomogramAveraging', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_STA,
                      label="Subtomogram averaging (StA)",
                      help='A StA previous run')

        form.addSection(label='combined rigid-body & elastic alignment')
        group = form.addGroup('Optical flow parameters', condition='Alignment_refine')
        group.addParam('N_GPU', params.IntParam, default=1, important=True, allowsNull=True,
                              label = 'Parallel processes on GPU',
                              help='This parameter indicates the number of volumes that will be processed in parallel'
                                   ' (independently). The more powerful your GPU, the higher the number you can choose.')
        group.addParam('GPU_list', params.NumericRangeParam,
                       label="GPU id(s)",
                       help='Select the GPU id(s) that will be used for optical flow calculation.'
                            'Examples: '
                            'You can select a list like 0-4, and it will take the GPUs 0 1 2 3 4'
                            'You can also combine different selections like 1, 3-5 and it will take 1, 3, 4, 5',
                       default='0')
        group.addParam('pyr_scale', params.FloatParam, default=0.5,
                      label='pyr_scale', allowsNull=True,
                       help='parameter specifying the image scale to build pyramids for each image (pyr_scale < 1). '
                            'A classic pyramid is of generally 0.5 scale, every new layer added, it is halved to the previous one.')
        group.addParam('levels', params.IntParam, default=4, allowsNull=True,
                      label='levels',
                      help='levels=1 says, there are no extra layers (only the initial image).'
                           ' It is the number of pyramid layers including the first image.'
                           ' The coarsest possible pyramid level is 32x32x32 voxels')
        group.addParam('winsize', params.IntParam, default=10, allowsNull=True,
                      label='winsize',
                      help='It is the averaging window size, larger the size, the more robust the algorithm is to noise, '
                           'and provide fast motion detection, though gives blurred motion fields.')
        group.addParam('iterations', params.IntParam, default=10, allowsNull=True,
                      label='iterations',
                      help='Number of iterations to be performed at each pyramid level. It refines the optical flow'
                           ' accuracy at each scale and allows accounting for larger displacements.')
        group.addParam('poly_n', params.IntParam, default=5, allowsNull=True,
                      label='poly_n',
                      help='Size of the pixel neighborhood used to find polynomial expansion in each pixel;'
                           ' larger values mean that the image will be approximated with smoother surfaces,'
                           ' yielding more robust algorithm and more blurred motion field, typically poly_n = 5 or 7.')
        group.addParam('poly_sigma', params.FloatParam, default=1.2,
                      label='poly_sigma',
                      help='Standard deviation of the Gaussian that is used to smooth derivatives used as '
                           'a basis for the polynomial expansion; for poly_n = 5, you can set poly_sigma = 1.2,'
                           ' for poly_n = 7, a good value would be poly_sigma = 1.5.')
        # This flag can be added later (when the Optical flow library is updated to include it)
        group.addHidden('flags', params.IntParam, default=0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='flags',
                      help='flag to pass for the optical flow')
        group.addHidden('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        group.addHidden('factor2', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor2',
                      help='this factor will be multiplied by the gray levels of the reference')
        group = form.addGroup('rigid-body alignment (refinement)', condition='Alignment_refine',)
        group.addParam('frm_freq', params.FloatParam, default=0.25,
                      label='Maximum cross correlation frequency',
                      help='The normalized frequency should be between 0 and 0.5 '
                           'The more it is, the bigger the search frequency is, the more time it demands, '
                           'keeping it as default is recommended.')
        group.addParam('frm_maxshift', params.IntParam, default=4,
                      label='Maximum shift for rigid body refinement (in pixels)',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'It represents the maximum distance searched in x,y and z directions.')

        form.addParallelSection(threads=0, mpi=multiprocessing.cpu_count()//2-1)


    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        # Make a new metadata fila that contains all the rigid-body information with updated names:
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('prepareMetaData')

        N = 0
        if (self.Alignment_refine.get()):
            N = self.NumOfIters.get()
            for i in range(1, N+1):
                makePath(self._getExtraPath() + '/optical_flows_' + str(i))
                if(self.FillWedge.get()):
                    self._insertFunctionStep('fillMissingWedge', i)
                self._insertFunctionStep('applyAlignment',i)
                self._insertFunctionStep('calculateOpticalFlows',i)
                self._insertFunctionStep('warpByFlow',i)
                self._insertFunctionStep('refineAlignment',i)
                self._insertFunctionStep('combineRefinedAlignment',i)
                self._insertFunctionStep('calculateNewAverage',i)

        if (self.FillWedge.get()):
            self._insertFunctionStep('fillMissingWedge', N+1)
        self._insertFunctionStep('applyAlignment', N+1)
        if self.Alignment_refine.get():
            self._insertFunctionStep('createOutputStep', N)
        else:
            self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))


    def prepareMetaData(self):
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
        # Sorting here To avoid future problems
        mdImgs.sort()
        mdImgs.write(self.imgsFn)


    def fillMissingWedge(self, num):
        tempdir = self._getTmpPath()
        # If this is the first iteration, then we have to use the starting metadata and subtomogram average
        if num == 1:
            imgFn = self.imgsFn
            StartingReference = self.StartingReference.get()
            ReferenceVolume = self.ReferenceVolume.get()
            if StartingReference == REFERENCE_STA:
                STAVolume = self.STAVolume.get().getFileName()
            else:
                STAVolume = ReferenceVolume
            # in case the reference is in MRC format:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')
            params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
            runProgram('xmipp_image_convert', params)

            # if there is a mask, then apply it:
            if (self.applyMask.get()):
                maskfn = self.Mask.get().getFileName()
                params = '-i ' + path_vol0 + ' -o ' + path_vol0 + ' --mult ' + maskfn
                runProgram('xmipp_image_operate', params)

        # Otherwise, we use the last itration of the combined refined alignment and the last average reached
        else:
            imgFn = self._getExtraPath('combined_'+str(num-1)+'.xmd')
            STAVolume = self._getExtraPath('reference' + str(num) + '.spi')
            # If this is not the first itration, remove the missing wedge filled data
            print('keep files options is ',self.KeepFiles.get())
            if(not(self.KeepFiles.get())):
                cleanPath(self._getExtraPath() + '/mw_filled_' + str(num-1))


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
        runProgram('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
        # Up to here, the missing wedge is created (this can be checked on the disk
        # to see if the missing wedge corresponds or not to the data)

        mdImgs = md.MetaData(imgFn)
        new_imgPath = self._getExtraPath() + '/mw_filled_' + str(num) + '/'
        makePath(new_imgPath)
        # Missing wedge filling now
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/mw_filled_' + str(num) + '/'
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            runProgram('xmipp_image_convert', params)
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
            if self.getAngleY()==90:
                params += '--inverse'
            # print('xmipp_transform_geometry',params)
            runProgram('xmipp_transform_geometry', params)
            if self.getAngleY() == 90:
                params = '-i ' + tempdir + '/temp.vol -o ' + tempdir + '/temp.vol '
                params += '--rotate_volume euler 0 -90 0 '
                # print('xmipp_transform_geometry',params)
                runProgram('xmipp_transform_geometry', params)
            # Now the STA is aligned, add the missing wedge region to the subtomogram:
            v = ImageHandler().read(new_imgPath).getData()
            I = fft(v)
            I = fftshift(I)
            v_ave = ImageHandler().read(tempdir + '/temp.vol').getData()
            Iave = fft(v_ave)
            Iave = fftshift((Iave))
            Mask = ImageHandler().read(fnmask).getData()

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
            if objId == 1:
                v_ave = ImageHandler().read(tempdir + '/temp.vol').getData()
                save_volume(v_ave, self._getExtraPath('aligned_average_with_first_volume.spi'))

        mdImgs.write(self._getExtraPath('MWFilled_' + str(num) + '.xmd'))


    def applyAlignment(self,num):
        makePath(self._getExtraPath()+'/aligned_'+str(num))
        tempdir = self._getTmpPath()

        # The aligned subtomograms are either the missing wedge filled or not according to the user choice
        if (self.FillWedge.get()):
            mdImgs = md.MetaData(self._getExtraPath('MWFilled_' + str(num) + '.xmd'))
        else:
            if num == 1:
                mdImgs = md.MetaData(self.imgsFn)
            else:
                mdImgs = md.MetaData(self._getExtraPath('combined_' + str(num - 1) + '.xmd'))

        if(num != 1):
            # If this is not the first itration, remove the previously aligned data
            if (not (self.KeepFiles.get())):
                cleanPath(self._getExtraPath() + '/aligned_' + str(num - 1))


        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath() + '/aligned_'+str(num) + '/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))

            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            # When we compensate for the missing wedge our software (FRM) doesn't have the same convention as XMIPP
            # So we have to rotate by the 90 degrees and use the software in inverse order, therefore you will find
            # a rotation of the subtomogram by 90 degrees and toogling the flag --inverse in xmipp_transform_geometry
            if self.getAngleY() == 90:
                params += '--rotate_volume euler 0 90 0 '
            else: # only to convert to spider in case it is something else (MRC for example)
                params += '--rotate_volume euler 0 0 0 '
            runProgram('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            if self.getAngleY() == 0:
                params += ' --inverse '

            runProgram('xmipp_transform_geometry', params)

            # if there is a mask, then apply it:
            if(self.applyMask.get()):
                maskfn = self.Mask.get().getFileName()
                params = '-i ' + new_imgPath + ' -o ' + new_imgPath + ' --mult ' + maskfn
                runProgram('xmipp_image_operate', params)

        self.fnaligned = self._getExtraPath('volumes_aligned_'+str(num)+'.xmd')
        mdImgs.write(self.fnaligned)


    def calculateOpticalFlows(self, num):
        tempdir = self._getTmpPath()
        imgFn = self._getExtraPath('volumes_aligned_'+str(num)+'.xmd')

        # in case it is the first iteration we only need the reference volume (metadata has to be for aligned volumes)
        if num == 1:
            StartingReference = self.StartingReference.get()
            ReferenceVolume = self.ReferenceVolume.get()
            if StartingReference == REFERENCE_STA:
                STAVolume = self.STAVolume.get().getFileName()
            else:
                STAVolume = ReferenceVolume

            # just in case the reference is in MRC format:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')
            params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
            runProgram('xmipp_image_convert', params)

            # if there is a mask, then apply it:
            if (self.applyMask.get()):
                maskfn = self.Mask.get().getFileName()
                params = '-i ' + path_vol0 + ' -o ' + path_vol0 + ' --mult ' + maskfn
                runProgram('xmipp_image_operate', params)

        else:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')
            # If this is not the first itration, remove the previous optical flows
            if(not(self.KeepFiles.get())):
                cleanPath(self._getExtraPath() + '/optical_flows_' + str(num - 1))

        pyr_scale = self.pyr_scale.get()
        levels = self.levels.get()
        iterations = self.iterations.get()
        winsize = self.winsize.get()
        poly_n = self.poly_n.get()
        poly_sigma = self.poly_sigma.get()
        flags = self.flags.get()
        factor1 = self.factor1.get()
        factor2 = self.factor2.get()

        mdImgs = md.MetaData(imgFn)
        of_root = self._getExtraPath() + '/optical_flows_' + str(num) + '/'

        # This is a spherical mask with maximum radius
        mask_size = int(self.getVolumeDimesion()//2)
        # Parallel processing (finding multiple optical flows at the same time)
        GPUids = np.array(getListFromRangeString(self.GPU_list.get()))
        gpu_ps = np.tile(GPUids, mdImgs.size())
        global segment
        def segment(objId):
            gpu_p = gpu_ps[objId-1]
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # getting a copy converted to spider format to solve the problem with stacks or mrc files
            tmp = self._getTmpPath('tmp_' + str(objId) + '.spi')
            runProgram('xmipp_image_convert', '-i ' + imgPath + ' -o ' + tmp + ' --type vol')

            print('processing optical flow for volume ', objId)
            path_flowx = of_root + str(objId).zfill(6) + '_opflowx.spi'
            path_flowy = of_root + str(objId).zfill(6) + '_opflowy.spi'
            path_flowz = of_root + str(objId).zfill(6) + '_opflowz.spi'
            path_vol_i = tmp
            if (isfile(path_flowx)):
                return
            else:
                args = " %s %s %f %d %d %d %d %f %d %d %s %s %s %d" % (path_vol0, path_vol_i, pyr_scale, levels, winsize,
                                                                   iterations, poly_n, poly_sigma, factor1, factor2,
                                                                   path_flowx, path_flowy, path_flowz, gpu_p)
                script_path = continuousflex.__path__[0] + '/protocols/utilities/optflow_run.py'
                command = "python " + script_path + args
                command = Plugin.getContinuousFlexCmd(command)
                check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                           env=None, cwd=None)

                arg_x = "-i %s  --mask circular -%d --substitute 0  -o %s" % (path_flowx, mask_size, path_flowx)
                arg_y = "-i %s  --mask circular -%d --substitute 0  -o %s" % (path_flowy, mask_size, path_flowy)
                arg_z = "-i %s  --mask circular -%d --substitute 0  -o %s" % (path_flowz, mask_size, path_flowz)
                runProgram('xmipp_transform_mask', arg_x)
                runProgram('xmipp_transform_mask', arg_y)
                runProgram('xmipp_transform_mask', arg_z)

        # Running the multiple processing:
        ps = [objId for objId in mdImgs]
        Parallel(n_jobs=self.N_GPU.get(), backend="multiprocessing")(delayed(segment)(p) for p in ps)


    def warpByFlow(self, num):
        makePath(self._getExtraPath() + '/estimated_volumes_' + str(num))
        if num != 1:
            if(not(self.KeepFiles.get())):
                cleanPath(self._getExtraPath() + '/estimated_volumes_' + str(num-1))
        estVol_root = self._getExtraPath() + '/estimated_volumes_' + str(num) + '/'
        reference = ImageHandler().read(self._getExtraPath('reference' + str(num) + '.spi')).getData()
        # recount the number of volumes:
        imgFn = self.imgsFn
        mdImgs = md.MetaData(imgFn)

        N = 0
        for objId in mdImgs:
            N += 1

        ref_dump = self._getTmpPath('ref_dump.pkl')
        joblib.dump(reference, ref_dump)

        mdWarped = md.MetaData()
        for i in range(1, N + 1):
            print('Warping a copy of the reference volume by the optical flow ', i)
            flow_i = self.read_optical_flow_by_number(i, op_path=self._getExtraPath() + '/optical_flows_' + str(num) + '/')
            # dumping the optical flow to use it in an outside script
            flow_dump = self._getTmpPath('flow_dump.pkl')
            joblib.dump(flow_i, flow_dump)
            warped_path_i = estVol_root + str(i).zfill(6) + '.spi'
            args = " %s %s %s" % (ref_dump, flow_dump, warped_path_i)
            script_path = continuousflex.__path__[0] + '/protocols/utilities/optflow_warp.py'
            command = "python " + script_path + args
            command = Plugin.getContinuousFlexCmd(command)
            check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                       env=None, cwd=None)
            mdWarped.setValue(md.MDL_IMAGE, warped_path_i, mdWarped.addObject())
            mdWarped.setValue(md.MDL_ITEM_ID, i, i)
        warpedVolFn = self._getExtraPath('warped_volumes_' + str(num) + '.xmd')
        mdWarped.write(warpedVolFn)


    def refineAlignment(self, num):
        imgFn = self._getExtraPath('warped_volumes_' + str(num) + '.xmd')
        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()
        result = self._getExtraPath('refinement_'+str(num)+'.xmd')
        reference = self._getExtraPath('reference' + str(num) + '.spi')
        tempdir = self._getTmpPath()
        args = "-i %(imgFn)s -o %(result)s --odir %(tempdir)s --resume --ref %(reference)s" \
               " --frm_parameters %(frm_freq)f %(frm_maxshift)d "

        self.runJob("xmipp_volumeset_align", args % locals(),
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

        mdImgs = md.MetaData(result)
        inputSet = md.MetaData(imgFn)
        # setting item_id (lost due to mpi) then sorting
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # Conside the index is the id in the input set
            for objId2 in inputSet:
                NewImgPath = inputSet.getValue(md.MDL_IMAGE, objId2)
                if (NewImgPath == imgPath):
                    target_ID = inputSet.getValue(md.MDL_ITEM_ID, objId2)
                    break
            mdImgs.setValue(md.MDL_ITEM_ID, target_ID, objId)
        mdImgs.sort()
        mdImgs.write(result)


    def combineRefinedAlignment(self, num):
        # 1- read both metadata (before and after refinment)
        if num == 1:
            MD_original = md.MetaData(self.imgsFn)
        else:
            MD_original = md.MetaData(self._getExtraPath('combined_'+str(num-1)+'.xmd'))

        MD_refined = md.MetaData(self._getExtraPath('refinement_'+str(num)+'.xmd'))
        # This metadata will be populated and saved
        MD_combined = md.MetaData()
        # Iteratively:
        # 2- find the transformation matrices
        for objId in MD_original:
            rot_o = MD_original.getValue(md.MDL_ANGLE_ROT, objId)
            tilt_o = MD_original.getValue(md.MDL_ANGLE_TILT, objId)
            psi_o = MD_original.getValue(md.MDL_ANGLE_PSI, objId)
            shiftx_o = MD_original.getValue(md.MDL_SHIFT_X, objId)
            shifty_o = MD_original.getValue(md.MDL_SHIFT_Y, objId)
            shiftz_o = MD_original.getValue(md.MDL_SHIFT_Z, objId)

            rot_r = MD_refined.getValue(md.MDL_ANGLE_ROT, objId)
            tilt_r = MD_refined.getValue(md.MDL_ANGLE_TILT, objId)
            psi_r = MD_refined.getValue(md.MDL_ANGLE_PSI, objId)
            shiftx_r = MD_refined.getValue(md.MDL_SHIFT_X, objId)
            shifty_r = MD_refined.getValue(md.MDL_SHIFT_Y, objId)
            shiftz_r = MD_refined.getValue(md.MDL_SHIFT_Z, objId)

            T_o = eulerAngles2matrix(rot_o, tilt_o, psi_o, shiftx_o, shifty_o, shiftz_o)
            T_r = eulerAngles2matrix(rot_r, tilt_r, psi_r, shiftx_r, shifty_r, shiftz_r)

            # 3- multiply the matrices
            if self.getAngleY() == 90:
                # In this case the refinement matrix should be inverted (because the refined alignment does not have
                # missing wedge correction)
                T_r_inv = np.linalg.inv(T_r)
                T = np.matmul(T_r_inv,T_o)
            else:
                # In this case the refinement matrix should be used as it is (as for both the previous and refined do not
                # have missing wedge correction)
                T = np.matmul(T_o, T_r)

            rot_i, tilt_i, psi_i, x_i, y_i, z_i = matrix2eulerAngles(T)

            # Populate the metadata
            name_i = MD_original.getValue(md.MDL_IMAGE, objId)
            MD_combined.setValue(md.MDL_IMAGE, name_i, MD_combined.addObject())
            MD_combined.setValue(md.MDL_ANGLE_ROT, rot_i, objId)
            MD_combined.setValue(md.MDL_ANGLE_TILT, tilt_i, objId)
            MD_combined.setValue(md.MDL_ANGLE_PSI, psi_i, objId)
            MD_combined.setValue(md.MDL_SHIFT_X, float(x_i), objId)
            MD_combined.setValue(md.MDL_SHIFT_Y, float(y_i), objId)
            MD_combined.setValue(md.MDL_SHIFT_Z, float(z_i), objId)
            if self.getAngleY() == 90:
                MD_combined.setValue(md.MDL_ANGLE_Y, 90.0, objId)
            else:
                MD_combined.setValue(md.MDL_ANGLE_Y, 0.0, objId)
            MD_combined.setValue(md.MDL_ITEM_ID, objId, objId)
        # Save the metadata
        MD_combined.write(self._getExtraPath('combined_'+str(num)+'.xmd'))


    def calculateNewAverage(self, num):
        # The flag will be used to know which alignment (missing wedge or without missing wedge) will be followed
        flag = self.getAngleY() == 90

        volumesMd = self._getExtraPath('combined_'+str(num)+'.xmd')
        mdVols = md.MetaData(volumesMd)

        counter = 0 # this is used to find the sum/number_of_volumes
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

            # The new reference is for the next iteration
            outputVol = self._getExtraPath('reference' + str(num+1) + '.spi')
            tempVol = self._getExtraPath('temp.vol')
            extra = self._getExtraPath()

            if flag == 0 :
                params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            else:
                # First got to rotate each volume 90 degrees about the y axis, align it, then sum it
                params = '-i %(imgPath)s -o %(tempVol)s --rotate_volume euler 0 90 0' % locals()
                runProgram('xmipp_transform_geometry', params)
                params = '-i %(tempVol)s -o %(tempVol)s --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s ' % locals()

            runProgram('xmipp_transform_geometry', params)

            if counter == 1 :
                os.system("mv %(tempVol)s %(outputVol)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(outputVol)s -o %(outputVol)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(outputVol)s --divide %(counter)s -o %(outputVol)s ' % locals()
        runProgram('xmipp_image_operate', params)

        # if there is a mask, then apply it:
        if (self.applyMask.get()):
            maskfn = self.Mask.get().getFileName()
            params = '-i ' + outputVol + ' -o ' + outputVol + ' --mult ' + maskfn
            runProgram('xmipp_image_operate', params)

        os.system("rm -f %(tempVol)s" % locals())


    def createOutputStep(self, num =0):
        out_mdfn = self._getExtraPath('volumes_aligned_' + str(num + 1) + '.xmd')
        partSet = self._createSetOfVolumes('aligned')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        # now creating the output of the final average if there is a refinement (otherwise no need as it is unchanged):
        if (self.Alignment_refine.get()):
            outvolume = Volume()
            outvolume.setSamplingRate((self.inputVolumes.get().getSamplingRate()))
            outvolume.setFileName(self._getExtraPath('reference' + str(num+1) + '.spi'))
            self._defineOutputs(OutputVolumes=partSet, RefinedAverage=outvolume)
        else:
            self._defineOutputs(OutputVolumes=partSet)

    # --------------------------- UTILS functions --------------------------------------------
    def read_optical_flow(self, path_flowx, path_flowy, path_flowz):
        x = ImageHandler().read(path_flowx).getData()
        y = ImageHandler().read(path_flowy).getData()
        z = ImageHandler().read(path_flowz).getData()

        l = np.shape(x)
        # print(l)
        flow = np.zeros([3, l[0], l[1], l[2]])
        flow[0, :, :, :] = x
        flow[1, :, :, :] = y
        flow[2, :, :, :] = z
        return flow

    def read_optical_flow_by_number(self, num, op_path = None):
        if op_path is None:
            op_path = self._getExtraPath() + '/optical_flows/'
        path_flowx = op_path + str(num).zfill(6) + '_opflowx.spi'
        path_flowy = op_path + str(num).zfill(6) + '_opflowy.spi'
        path_flowz = op_path + str(num).zfill(6) + '_opflowz.spi'
        flow = self.read_optical_flow(path_flowx, path_flowy, path_flowz)
        return flow

    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    # AngleY should never be 90 from now on. However, it can stay here in order someone imports an old STA alignment
    def getAngleY(self):
        AlignmentParameters = self.AlignmentParameters.get()
        MetaDataFile = self.MetaDataFile.get()
        if AlignmentParameters == REFERENCE_STA:
            MetaDataFile = self.MetaDataSTA.get()._getExtraPath('final_md.xmd')
        try:
            mdFile = md.MetaData(MetaDataFile)
            angleY = mdFile.getValue(md.MDL_ANGLE_Y, 1)
            if angleY is None:
                angleY = 0
        except:
            angleY = 0

        return int(angleY)

    def getVolumeDimesion(self):
        return self.inputVolumes.get().getDimensions()[0]

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022tomoflow','harastani2022continuousflex']

    def _methods(self):
        pass
