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
import joblib
from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath, createLink
from continuousflex.protocols.utilities.spider_files3 import save_volume
import sys
from pyworkflow.utils import getListFromRangeString
from os.path import isfile
from joblib import Parallel, delayed
import continuousflex
from subprocess import check_call
from pwem.utils import runProgram
from pwem.emlib.image import ImageHandler
import numpy as np
from continuousflex import Plugin

REFERENCE_EXT = 0
REFERENCE_STA = 1

IMPORT_FLOWS = 0
FIND_FLOWS = 1

class FlexProtHeteroFlow(ProtAnalysis3D):
    """ Protocol for TomoFlow. """
    _label = 'tomoflow protocol'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Settings')
        group = form.addGroup('Choose what processes you want to perform:')
        group.addParam('copy_opflows', params.EnumParam,
                      choices=['Import optical flows from the last refinement iteration and analyze the heterogeneity',
                               'Find optical flows for a set of ALIGNED volumes and analyze the heterogeneity'],
                      default=IMPORT_FLOWS,
                      label='Optical flows to analyze', display=params.EnumParam.DISPLAY_COMBO,
                       help='You can choose to find the optical flows for a set of volumes or to import'
                            ' precalculated optical flows from a refinement protocol previous run in the project'
                            ' workspace')
        group = form.addGroup('Protocol of: Missing wedge correction and combined rigid-body and elastic alignment',
                              condition='copy_opflows==%d'% IMPORT_FLOWS)
        group.addParam('refinementProt', params.PointerParam, pointerClass='FlexProtRefineSubtomoAlign',
                       label='Point to the refinement protocol', allowsNull=True)

        group = form.addGroup('Input', condition='copy_opflows==%d' % FIND_FLOWS)
        group.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes', allowsNull=True,
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        group.addParam('StartingReference', params.EnumParam,
                      choices=['From an external volume file', 'Select a volume'],
                      default=REFERENCE_STA,
                      label='Reference volume', display=params.EnumParam.DISPLAY_COMBO,
                      help='Either an external volume file or a subtomogram average')
        group.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Reference volume",
                      help='Choose a reference, typically from a STA previous run')
        group.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Selected volume",
                      help='Choose a reference, typically from a STA previous run')
        group.addParam('WarpAndEstimate', params.BooleanParam,
                      expertLevel=params.LEVEL_ADVANCED,
                      default=True,
                      label='Save a warped version of the reference for each input volume?',
                      help='This will create a set of volumes, representing the fitted version of the input volumes, '
                           'using the calculated optical flows, and calculate the cross correlation, mean square '
                           'distance and the mean absolute distance between the input volumes and estimated volumes')
        form.addSection(label='3D OpticalFLow parameters')
        group = form.addGroup('Optical flows', condition='copy_opflows==%d' % FIND_FLOWS)
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
        group.addHidden('flags', params.IntParam, default=0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='flags',
                      help='flag to pass for the optical flow')
        group.addHidden('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='gray scale factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        group.addHidden('factor2', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='gray scale factor2',
                      help='this factor will be multiplied by the gray levels of the reference')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')

        if(self.copy_opflows.get()==FIND_FLOWS):
            makePath(self._getExtraPath() + '/optical_flows')
            self._insertFunctionStep('convertInputStep')
            self._insertFunctionStep('doAlignmentStep')
        else:
            self._insertFunctionStep('copyOpticalFlows')
        self._insertFunctionStep('findCorrelationMatrix')
        if (self.WarpAndEstimate.get()):
            self._insertFunctionStep('warpByFlow')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)

    def doAlignmentStep(self):
        imgFn = self.imgsFn
        StartingReference = self.StartingReference.get()
        ReferenceVolume = self.ReferenceVolume.get()

        if StartingReference == REFERENCE_STA:
            STAVolume = self.STAVolume.get().getFileName()
        else:
            STAVolume = ReferenceVolume
        # just in case the reference is in MRC format:
        path_vol0 = self._getExtraPath('reference.spi')
        params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
        runProgram('xmipp_image_convert', params)

        pyr_scale = self.pyr_scale.get()
        levels = self.levels.get()
        iterations = self.iterations.get()
        winsize = self.winsize.get()
        poly_n = self.poly_n.get()
        poly_sigma = self.poly_sigma.get()
        # TODO: the factor1 and 2 can be any value as long as we are using the subtomogram average (gray level values
        # are similar. It is not sure if we use an external reference what this should be! This could be normalized in
        #  future
        flags = self.flags.get()
        factor1 = self.factor1.get()
        factor2 = self.factor2.get()

        mdImgs = md.MetaData(imgFn)
        of_root = self._getExtraPath() + '/optical_flows/'

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

    def findCorrelationMatrix(self):
        imgFn = self.imgsFn
        mdImgs = md.MetaData(imgFn)
        N = 0
        for objId in mdImgs:
            N += 1
        metric_mat = np.zeros([N, N])
        for i in range(1, N + 1):
            print('finding the correlation matrix row ', i)
            flowi = self.read_optical_flow_by_number(i)
            for j in range(i, N + 1):
                print('        column', j)
                flowj = self.read_optical_flow_by_number(j)
                metric_mat[i - 1, j - 1] = metric_mat[j - 1, i - 1] =  self.metric_opflow_vols(flowi, flowj)

        correlation_matrix = self._getExtraPath('data.csv')
        np.savetxt(correlation_matrix, metric_mat, delimiter=',')

    def copyOpticalFlows(self):
        # In this case we get from the refinment protocol the optical flows and the reference
        N = self.refinementProt.get().NumOfIters.get()
        self.imgsFn = self.refinementProt.get()._getExtraPath('volumes_aligned_'+str(N+1)+'.xmd')
        createLink(self.refinementProt.get()._getExtraPath() + '/optical_flows_' + str(N),
                   self._getExtraPath() + '/optical_flows')
        createLink(self.refinementProt.get()._getExtraPath('reference'+str(N+1)+'.spi'),
                   self._getExtraPath('reference.spi'))

    def warpByFlow(self):
        makePath(self._getExtraPath() + '/estimated_volumes')
        estVol_root = self._getExtraPath() + '/estimated_volumes/'
        reference_fn = self._getExtraPath('reference.spi')
        # reference = open_volume(reference_fn)
        reference = ImageHandler().read(reference_fn).getData()

        stat_mat_fn = self._getExtraPath('cc_msd_mad.txt')
        # recount the number of volumes:
        imgFn = self.imgsFn
        mdImgs = md.MetaData(imgFn)

        N = 0
        for objId in mdImgs:
            N += 1

        # dumping the reference volume to use it in an outside script
        ref_dump = self._getTmpPath('ref_dump.pkl')
        joblib.dump(reference, ref_dump)

        # TODO: this loop can be parallelized, but it is not too computationally demanding
        for i in range(1, N + 1):
            print('Warping a copy of the reference volume by the optical flow ', i)
            flow_i = np.float32(self.read_optical_flow_by_number(i))
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

        # Find a matrix of metrics (normalized cross correlation, mean square distance, mean absolute distance)
        stat_mat = np.zeros([N, 3])
        mdImgs = md.MetaData(imgFn)
        i = 0
        for objId in mdImgs:
            print('Finding the cross corrlation, mean square distance and mean absolute distance for volume ', i + 1)
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # getting a copy converted to spider format to solve the problem with stacks or mrc files
            tmp = self._getTmpPath('tmp.spi')
            runProgram('xmipp_image_convert', '-i ' + imgPath + ' -o ' + tmp + ' --type vol')
            # vol_i = open_volume(tmp)
            vol_i = ImageHandler().read(tmp).getData()

            warped_path_i = estVol_root + str(i + 1).zfill(6) + '.spi'
            warped_i = ImageHandler().read(warped_path_i).getData()
            # TODO: replace all the open_volume and save_volume by the ImageHandler() read and write
            # warped_i = open_volume(warped_path_i)
            stat_mat[i, 0] = self.ncc(warped_i,vol_i)
            stat_mat[i, 1] = self.vmsq(warped_i,vol_i)
            stat_mat[i, 2] = self.vmab(warped_i,vol_i)
            i += 1
        np.savetxt(stat_mat_fn, stat_mat)
        pass

    def createOutputStep(self):
        if (self.WarpAndEstimate.get()):
            # first making a metadata for the wrapped volumes:
            out_mdfn = self._getExtraPath('volumes_out.xmd')
            pattern = '"' + self._getExtraPath() + '/estimated_volumes/*.spi"'
            command = '-p ' + pattern + ' -o ' + out_mdfn
            runProgram('xmipp_metadata_selfile_create',command)
            # now creating the output set of volumes:
            partSet = self._createSetOfVolumes('Warped')
            xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
            if (self.copy_opflows.get() == FIND_FLOWS):
                partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            else:
                partSet.setSamplingRate(self.refinementProt.get().RefinedAverage.getSamplingRate())
            self._defineOutputs(WarpedRefByFlows=partSet)
            pass
        pass

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

    def read_optical_flow_by_number(self, num):
        op_path = self._getExtraPath() + '/optical_flows/'
        path_flowx = op_path + str(num).zfill(6) + '_opflowx.spi'
        path_flowy = op_path + str(num).zfill(6) + '_opflowy.spi'
        path_flowz = op_path + str(num).zfill(6) + '_opflowz.spi'
        flow = self.read_optical_flow(path_flowx, path_flowy, path_flowz)
        return flow

    def metric_opflow_vols(self, flow1, flow2):
        # print(np.shape(flow1))
        reshaped_flow1 = np.reshape(flow1, [3, np.shape(flow1)[1] * np.shape(flow1)[2] * np.shape(flow1)[3]])
        reshaped_flow2 = np.reshape(flow2, [3, np.shape(flow2)[1] * np.shape(flow2)[2] * np.shape(flow2)[3]])
        metric = np.sum(np.multiply(reshaped_flow1, reshaped_flow2))
        return metric

    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    def normalize(self, v):
        """Normalize the data.
        @param v: input volume.
        @return: Normalized volume.
        """
        m = np.mean(v)
        v = v - m
        s = np.std(v)
        v = v / s
        return v

    def ncc(self, v1, v2):
        """Compute the Normalized Cross Correlation between the two volumes.
        @param v1: volume 1.
        @param v2: volume 2.
        @return: NCC.
        """
        vv1 = self.normalize(v1)
        vv2 = self.normalize(v2)
        score = np.sum(vv1 * vv2) / vv1.size
        return score

    def vmsq(self, v1, v2):
        """Compute the normalized mean square distance between the two volumes
        :param v1: volume1
        :param v2: volume2
        :return: score for mean square diff
        """
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(np.ndarray.flatten(v1), np.ndarray.flatten(v2)) / mean_squared_error(
            np.ndarray.flatten(v1),
            np.zeros([np.size(np.ndarray.flatten(v1))]))
        return score

    def vmab(self, v1, v2):
        """Compute the normalized mean absolute distance between the two volumes
        :param v1: volume1
        :param v2: volume2
        :return: score for mean square diff
        """
        from sklearn.metrics import mean_absolute_error
        score = mean_absolute_error(np.ndarray.flatten(v1), np.ndarray.flatten(v2)) / mean_absolute_error(
            np.ndarray.flatten(v1),
            np.zeros([np.size(np.ndarray.flatten(v1))]))
        return score

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