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
    """ Protocol for subtomogram missingwedge filling. """
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
