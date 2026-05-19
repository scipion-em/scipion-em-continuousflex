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

from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath
from os.path import basename, isfile
from .utilities.spider_files3 import save_volume
from pyworkflow.utils import replaceBaseExt
import numpy as np
from continuousflex.protocols.utilities.mwr_wrapper import mwr
from continuousflex.protocols.protocol_subtomograms_synthesize import FlexProtSynthesizeSubtomo
from pwem.utils import runProgram

METHOD_MCSFILL = 0


class FlexProtMissingWedgeRestoration(ProtAnalysis3D):
    """
    Restores missing information in subtomogram volumes affected by the missing wedge artifact
    generated during electron tomography data acquisition. The protocol aims to reduce anisotropic
    distortions and recover a more complete representation of the underlying structure, improving
    the interpretability of tomographic reconstructions.

    AI Generated:

    Missing Wedge Restoration (FlexProtMissingWedgeRestoration) - User Manual
        Overview

        The Missing Wedge Restoration protocol is designed to compensate for one of the most common
        limitations in electron tomography: the incomplete angular sampling that occurs during tilt
        series acquisition. Because physical and experimental constraints prevent collecting images
        over the full angular range, reconstructed tomograms contain a region of missing information
        in Fourier space commonly referred to as the missing wedge.

        This missing information introduces directional artifacts, anisotropic resolution, elongation
        effects, and distortions that can complicate structural interpretation. The purpose of this
        protocol is to reduce the impact of these artifacts and generate volumes that more accurately
        represent the biological structures present in the sample.

        Biological Motivation

        In cryo-electron tomography, macromolecular complexes are often studied directly within their
        native cellular environment. Although this approach provides unique biological insight, the
        limited tilt range leads to incomplete sampling of structural information.

        The resulting missing wedge can affect particle classification, structural averaging, conformational
        analysis, and visualization. Features oriented along poorly sampled directions may appear blurred,
        elongated, or partially absent. Restoration methods seek to alleviate these limitations and
        provide a more balanced representation of structural details.

        For biological users, missing wedge correction can improve the reliability of downstream analyses,
        particularly when comparing conformational states, identifying structural features, or studying
        heterogeneous populations of macromolecular assemblies.

        Inputs and Experimental Parameters

        The protocol accepts one or multiple reconstructed volumes as input. These volumes are assumed
        to originate from electron tomography experiments where the angular acquisition range is known.

        The user must specify the lower and upper tilt angles used during data collection. These values
        define the region of Fourier space that was experimentally sampled and therefore determine the
        shape and extent of the missing wedge artifact affecting the reconstruction.

        Accurate tilt limits are important because they directly influence the restoration process.
        Whenever possible, users should provide values that match the actual acquisition conditions
        rather than idealized microscope settings.

        Monte Carlo Based Restoration

        The protocol employs a Monte Carlo based restoration strategy specifically designed to estimate
        plausible structural information within the missing wedge region. Rather than simply filtering
        the reconstruction, the method attempts to infer missing content while maintaining consistency
        with the experimentally observed data.

        This probabilistic approach is particularly attractive for tomographic datasets because it can
        model uncertainty within unsampled regions while preserving the information already supported
        by the measurements. The result is typically a more isotropic reconstruction with reduced
        directional artifacts.

        Since the missing information is fundamentally unknown, restored regions should be interpreted
        as statistically plausible estimates rather than direct experimental observations. Biological
        conclusions should therefore rely on features that remain consistent across the dataset and
        are supported by additional evidence whenever possible.

        Noise Modeling and Regularization

        The restoration process incorporates a noise parameter that controls the balance between
        preserving detail and enforcing smoothness. Larger values generally produce smoother volumes
        and stronger regularization, whereas smaller values preserve finer structural features.

        From a biological perspective, selecting an appropriate value depends on the quality of the
        tomographic reconstruction. Noisy datasets may benefit from stronger regularization, while
        high-quality reconstructions often allow more conservative settings that preserve subtle
        structural details.

        As with many restoration methods, excessive smoothing can suppress meaningful biological
        features, whereas insufficient regularization may leave residual artifacts. Testing multiple
        values and visually comparing the results is often beneficial.

        Iterative Sampling Parameters

        The protocol provides control over the number of restoration iterations and the length of the
        burn-in phase used during the sampling procedure. Increasing the number of iterations generally
        improves convergence and stability of the estimated solution but also increases computational
        cost.

        The burn-in phase represents an initial period during which intermediate estimates are discarded.
        This allows the restoration process to move away from its starting conditions before generating
        the final solution. For most datasets, the default values provide a reasonable compromise between
        computational efficiency and restoration quality.

        Advanced users may adjust these parameters when working with particularly noisy datasets or when
        pursuing highly quantitative analyses.

        Outputs and Their Interpretation

        The protocol generates a restored set of volumes in which the effects of the missing wedge have
        been reduced. These restored volumes can be used in subsequent stages of analysis, including
        classification, averaging, flexibility studies, dimensionality reduction, and structural
        interpretation.

        Restoration often improves visual continuity and isotropy within the reconstructed structures.
        Features that were previously obscured by directional artifacts may become easier to identify
        and analyze. Nevertheless, users should remember that restoration cannot recreate the exact
        missing experimental information and therefore does not replace careful biological validation.

        Practical Recommendations

        Before applying restoration, it is advisable to verify that the tilt limits accurately reflect
        the acquisition geometry. Incorrect angular ranges may lead to suboptimal correction and could
        introduce additional artifacts.

        Visual comparison between original and restored volumes is strongly recommended. Improvements
        should be assessed not only by appearance but also by consistency with known biological features
        and independent experimental evidence.

        When restored volumes are intended for downstream quantitative analyses, users should evaluate
        whether the restoration procedure alters measurements relevant to their specific biological
        questions.

        Final Perspective

        Missing wedge artifacts remain one of the major limitations of electron tomography. This protocol
        provides a dedicated framework for mitigating their impact through probabilistic restoration,
        helping researchers obtain more isotropic and biologically interpretable reconstructions. When
        applied carefully and interpreted appropriately, missing wedge restoration can substantially
        improve the quality and usefulness of tomographic datasets.
    """
    _label = 'missing wedge restoration'

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
                      choices=['MW restoration using monte carlo simulation'],
                      default=METHOD_MCSFILL,
                      label='Missing wedge (MW) correction method', display=params.EnumParam.DISPLAY_COMBO,
                      help=' The monte carlo method is an implementation of the method of E. Moebel & C. Kervrann')
        group2 = form.addGroup('MW restoration using monte carlo simulation',
                      condition='Method==%d' % METHOD_MCSFILL)
        group2.addParam('sigma_noise', params.FloatParam, default=0.2, allowsNull=True,
                       label='noise sigma', important= True,
                       help='estimated standard deviation of data noise '
                            'defines the strength of the processing (high value gives smooth images)')
        group2.addParam('T', params.IntParam, default=300, allowsNull=True,
                        label='number of iterations',
                        expertLevel=params.LEVEL_ADVANCED,
                        help='number of iterations (default: 300)')
        group2.addParam('Tb', params.IntParam, default=100, allowsNull=True,
                        label='length of the burn-in phase (Tb)',
                        expertLevel=params.LEVEL_ADVANCED,
                        help='First Tb samples are discarded (default: 100)')
        group2.addParam('beta', params.FloatParam, default=0.00004, allowsNull=True,
                        label='scale parameter (beta)',
                        expertLevel=params.LEVEL_ADVANCED,
                        help='scale parameter, affects the acceptance rate (default: 0.00004)')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/mw_filled')
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('doAlignmentStep_MCSFILL')
        self._insertFunctionStep('createOutputStep_MCSFILL')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        try:
            xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))
        except:
            mdF = md.MetaData()
            mdF.setValue(md.MDL_IMAGE, self.inputVolumes.get().getFileName(), mdF.addObject())
            mdF.write(self.imgsFn)
            pass

    def doAlignmentStep_MCSFILL(self):
        # get a copy of the input metadata unless if one volume is passed
        try:
            xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        except:
            pass
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
        runProgram('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
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
            runProgram('xmipp_image_convert', params)
            # perform the mwr:
            # in case the file exists (continuing or injecting)
            if (isfile(new_imgPath)):
                continue
            else:
                mwr(temp_path,fnmask,new_imgPath,sigma_noise,T,Tb,beta,True)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
        mdImgs.write(self.imgsFn)

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
        return ['harastani2022continuousflex','moebel2020monte']

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