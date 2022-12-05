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
    """ Protocol for subtomogram missingwedge restoration. """
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