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
from pyworkflow.utils.path import makePath, copyFile
from os.path import basename, isfile
from pyworkflow.utils import replaceBaseExt
from continuousflex.protocols.utilities.bm4d import bm4d
from pwem.utils import runProgram


REFERENCE_EXT = 0
REFERENCE_STA = 1

METHOD_BM4D = 0
METHOD_LOWPASS = 1

NOISE_GAUSS = 0
NOISE_RICE = 1

PROFILE_LC = 0
PROFILE_NP = 1
PROFILE_MP = 2


class FlexProtVolumeDenoise(ProtAnalysis3D):
    """
    Denoises three-dimensional volumes using advanced noise reduction
    techniques. The protocol improves volume quality by reducing unwanted
    noise while attempting to preserve biologically relevant structural
    information.

    AI Generated:

    Volume Denoise (FlexProtVolumeDenoise) - User Manual
        Overview

        The Volume Denoise protocol is designed to enhance the quality of
        three-dimensional volumes by reducing noise that may obscure
        meaningful structural features. In cryo-electron microscopy,
        subtomogram analysis, and related volumetric imaging workflows,
        noise is an unavoidable component of experimental data and can
        significantly affect visualization, interpretation, classification,
        and downstream computational analyses.

        The primary objective of this protocol is to improve the signal-to-
        noise ratio while maintaining the integrity of biologically relevant
        structures. By producing cleaner volumes, the protocol facilitates
        subsequent procedures such as averaging, flexible analysis,
        segmentation, structural comparison, and molecular interpretation.

        Inputs and General Workflow

        The protocol accepts either a single volume or a collection of
        volumes. This flexibility allows users to process individual
        reconstructions as well as large datasets generated during
        subtomogram averaging, classification, or conformational studies.

        Each input volume is processed independently using the selected
        denoising strategy. The resulting outputs preserve the identity and
        organization of the original dataset while providing improved image
        quality for further analysis.

        Choice of Denoising Method

        The protocol provides two alternative approaches for noise reduction,
        each suited to different scientific objectives and data conditions.

        The BM4D method is an advanced volumetric denoising strategy
        specifically designed for three-dimensional data. It is generally
        preferred when preserving subtle structural details is important.
        This approach is particularly valuable for cryo-EM and tomography
        datasets where signal levels are low and structural features may be
        difficult to distinguish from background noise.

        The Fourier low-pass filtering method is a simpler and computationally
        efficient approach. It attenuates high-frequency components that are
        frequently dominated by noise while retaining lower-frequency
        structural information. This method is useful for rapid exploratory
        analysis or for datasets where fine high-resolution information is
        not the primary focus.

        BM4D Noise Modeling

        When using BM4D, the protocol allows the user to specify the expected
        statistical behavior of the noise. Different noise distributions may
        better describe different imaging conditions, and selecting an
        appropriate model can improve denoising performance.

        The protocol also provides control over the estimated noise level.
        This parameter strongly influences the balance between noise removal
        and structural preservation. Lower values generally preserve more
        detail but may leave residual noise, whereas higher values produce
        smoother volumes at the risk of suppressing weak biological features.

        Several processing profiles are available to accommodate different
        computational and denoising requirements. These profiles allow users
        to adapt the method to the characteristics of their data and the
        desired level of noise suppression.

        An optional Wiener refinement stage may also be employed. This
        additional processing can improve denoising performance in some
        datasets by further enhancing signal recovery while maintaining
        structural consistency.

        Fourier Low-Pass Filtering

        The Fourier filtering approach removes high-frequency information
        beyond a selected cutoff frequency. Biologically, this operation can
        be interpreted as emphasizing large-scale structural organization
        while reducing small-scale fluctuations dominated by noise.

        The cutoff frequency determines the effective resolution retained in
        the processed volume. Lower cutoff values produce smoother volumes
        with stronger noise suppression, whereas higher values preserve more
        structural detail. A gradual transition region can also be applied to
        reduce filtering artifacts and generate more natural-looking results.

        Biological Considerations

        Denoising should always be performed with awareness of the biological
        question being addressed. Excessive noise reduction may remove weak
        but meaningful structural features, particularly in flexible regions,
        small domains, or low-occupancy conformations.

        For exploratory visualization and qualitative interpretation,
        stronger denoising may be acceptable. However, for quantitative
        analyses or studies involving subtle conformational differences,
        conservative processing is generally recommended to avoid introducing
        bias or obscuring genuine variability.

        In heterogeneous datasets, users should be especially cautious when
        comparing denoised volumes. Differences introduced by aggressive
        filtering may sometimes be mistaken for biological variation.

        Outputs and Their Interpretation

        The protocol produces a denoised volume or set of denoised volumes
        corresponding directly to the provided inputs. The outputs retain the
        original sampling characteristics while exhibiting reduced noise and
        improved visual clarity.

        These processed volumes can be used for visualization, classification,
        flexible analysis, segmentation, or as inputs to additional
        computational workflows. The denoised results should nevertheless be
        interpreted alongside the original data whenever critical biological
        conclusions are being drawn.

        Practical Recommendations

        For most cryo-EM and subtomogram analysis applications, BM4D is
        generally the preferred starting point because it provides strong
        noise reduction while preserving structural information more
        effectively than simple frequency filtering.

        Fourier low-pass filtering is useful for rapid preprocessing,
        visualization, or situations where computational simplicity is
        desired. It can also serve as an initial assessment tool before more
        advanced denoising methods are applied.

        Users should evaluate denoising results visually and, whenever
        possible, compare them against the original volumes to ensure that
        biologically meaningful features have not been inadvertently removed.

        Final Perspective

        Noise reduction is often a crucial step in volumetric structural
        biology workflows. Effective denoising can substantially improve the
        interpretability of experimental data and facilitate downstream
        analyses. The most reliable results are obtained when the denoising
        strategy is selected according to the characteristics of the dataset
        and the biological objectives of the study, balancing noise
        suppression with faithful preservation of structural information.
    """
    _label = 'volume denoise'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        form.addSection('Method')
        form.addParam('Method', params.EnumParam,
                      choices=['BM4D', 'Fourier lowpass filter'],
                      default=METHOD_BM4D,
                      label='Denoising Method', display=params.EnumParam.DISPLAY_COMBO,
                      help='Choose a method: BM4D or Fourier lowpass filter')
        group = form.addGroup('BM4D parameters', condition='Method==%d' % METHOD_BM4D)
        group.addParam('noiseType', params.EnumParam,
                      choices=['Gaussian', 'Rician'],
                      default=NOISE_GAUSS,
                      label='Noise distribution', display=params.EnumParam.DISPLAY_COMBO,
                      help='Noise distribution (either Gaussian or Rician)')
        group.addParam('sigma_choice', params.EnumParam,
                      choices=['Automatically estimate sigma', 'Set a value for sigma (recommended)'],
                      default=1,
                      label='Sigma choice', display=params.EnumParam.DISPLAY_COMBO,
                      help='Sigma is the standard deviation of data noise')
        group.addParam('sigma', params.FloatParam, default=0.2, allowsNull=True,
                      condition='sigma_choice==%d' % 1,
                      label='Sigma',
                      help='estimated standard deviation of data noise '
                           'defines the strength of the processing (high value gives smooth images)')
        group.addParam('profile', params.EnumParam,
                      choices=['low complexity profile', 'normal profile', 'modified profile (recommended)'],
                      default=PROFILE_MP,
                      label='Noise profile', display=params.EnumParam.DISPLAY_COMBO,
                      help='lc --> low complexity profile, '
                           ' np --> normal profile,'
                           ' mp --> modified profile')
        group.addParam('do_wiener', params.BooleanParam, allowsNull=True,
                      default=False,
                      label='Do wiener?',
                      help='Perform collaborative Wiener filtering')

        # Normalized frequencies ("digital frequencies")
        line = form.addLine('Frequency (normalized)',
                            condition='Method==%d' % METHOD_LOWPASS,
                            help='The cufoff frequency and raised coside width of the low pass filter.'
                                 ' For details: see "xmipp_transform_filter --fourier low_pass"')
        line.addHidden('lowFreqDig', params.DigFreqParam, default=0.00, allowsNull=True,
                        label='Lowest')
        line.addParam('highFreqDig', params.DigFreqParam, default=0.25, allowsNull=True,
                      label='Cutoff frequency (0 -> 0.5)')
        line.addParam('freqDecayDig', params.FloatParam, default=0.02, allowsNull=True,
                      label='Raised cosine width')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/filtered')
        self._insertFunctionStep('convertInputStep')
        if(self.Method.get()==METHOD_BM4D):
            self._insertFunctionStep('denoise_b4md')
        else:
            self._insertFunctionStep('filter_lowpass')
        self._insertFunctionStep('createOutputStep')
        pass

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        try:
            xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        except:
            mdF = md.MetaData()
            mdF.setValue(md.MDL_IMAGE, self.inputVolumes.get().getFileName(), mdF.addObject())
            mdF.write(self.imgsFn)
            pass

    def denoise_b4md(self):
        distribution = ''
        if self.noiseType.get() == NOISE_GAUSS:
            distribution = 'Gauss'
        else:
            distribution = 'Rice'
        sigma = self.sigma.get()
        profile = ''
        if self.profile.get()==PROFILE_LC:
            profile = 'lc'
        elif self.profile.get()==PROFILE_NP:
            profile = 'np'
        elif self.profile.get()==PROFILE_MP:
            profile = 'mp'
        else:
            exit()
        do_weiner = 0
        if self.do_wiener.get():
            do_weiner = 1


        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        # looping on all images and performing mwr
        mdImgs = md.MetaData(imgFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/filtered/'
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
                bm4d(temp_path, new_imgPath, distribution, sigma, profile, do_weiner)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
        mdImgs.write(self.imgsFn)


    def filter_lowpass(self):
        cutoff = self.highFreqDig.get()
        raisedw = self.freqDecayDig.get()

        imgFn = self.imgsFn
        # looping on all images and performing mwr
        mdImgs = md.MetaData(imgFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/filtered/'
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
                params = " -i " + temp_path + " -o " + new_imgPath
                params += " --fourier low_pass " + str(cutoff) + ' ' + str(raisedw)
                runProgram('xmipp_transform_filter', params)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
        mdImgs.write(self.imgsFn)

    def createOutputStep(self):
        partSet = self._createSetOfVolumes('filtered')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(filteredVolumes=partSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex']

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
