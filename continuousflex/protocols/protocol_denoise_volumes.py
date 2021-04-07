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
from continuousflex.protocols.utilities.bm4d import bm4d

REFERENCE_EXT = 0
REFERENCE_STA = 1

METHOD_BM4D = 0

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
                      choices=['bm4d'],
                      default=METHOD_BM4D,
                      label='Denoising Method', display=params.EnumParam.DISPLAY_COMBO,
                      help='Denoise using bm4d')
        form.addParam('noiseType', params.EnumParam,
                      choices=['Gaussian', 'Rician'],
                      default=NOISE_GAUSS,
                      label='Noise distribution', display=params.EnumParam.DISPLAY_COMBO,
                      help='Noise distribution (either Gaussian or Rician)')
        form.addParam('sigma_choice', params.EnumParam,
                      choices=['Automatically estimate sigma', 'Set a value for sigma'],
                      default=0,
                      label='Sigma choice', display=params.EnumParam.DISPLAY_COMBO,
                      help='Sigma is the standard deviation of data noise')
        form.addParam('sigma', params.FloatParam, default=0, allowsNull=True,
                      condition='sigma_choice==%d' % 1,
                      label='Sigma',
                      help='estimated standard deviation of data noise '
                           'defines the strength of the processing (high value gives smooth images)')
        form.addParam('profile', params.EnumParam,
                      choices=['lc', 'np', 'mp'],
                      default=PROFILE_MP,
                      label='Noise profile', display=params.EnumParam.DISPLAY_COMBO,
                      help='lc --> low complexity profile, '
                           ' np --> normal profile'
                           ' mp --> modified profile')
        form.addParam('do_wiener', params.BooleanParam, allowsNull=True,
                      default=False,
                      label='Do wiener?',
                      help='Perform collaborative Wiener filtering')

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/filtered')
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('denoise_b4md')
        self._insertFunctionStep('createOutputStep')
        pass

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)


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
            self.runJob('xmipp_image_convert', params)
            # perform the mwr:
            # in case the file exists (continuing or injecting)
            if (isfile(new_imgPath)):
                continue
            else:
                bm4d(temp_path, new_imgPath, distribution, sigma, profile, do_weiner)
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
