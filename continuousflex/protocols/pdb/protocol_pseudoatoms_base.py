# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Slavica Jonic                (jonic@impmc.upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Jan 2014
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

from os.path import basename, join, abspath
import os

from pyworkflow.object import String
from pyworkflow.utils import cleanPattern, moveFile
from pyworkflow.protocol.params import EnumParam, PointerParam, FloatParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem.protocols import Prot3D #this is not an error
from pwem.viewers.viewer_chimera import Chimera
from xmipp3.convert import getImageLocation
from pwem import Domain


NMA_MASK_NONE = 0
NMA_MASK_THRE = 1
NMA_MASK_FILE = 2


class FlexProtConvertToPseudoAtomsBase(Prot3D):
    # --------------------------- DEFINE param functions --------------------
    def _defineParams(self, form):
        form.addParam('maskMode', EnumParam,
                      choices=['none', 'threshold', 'file'],
                      default=NMA_MASK_NONE,
                      label='Mask mode', display=EnumParam.DISPLAY_COMBO,
                      help='Mask to remove the background noise in the volume. \n'
			   'If the volume was masked outside of this program, no need for masking here.')
        form.addParam('maskThreshold', FloatParam, default=0.01,
                      condition='maskMode==%d' % NMA_MASK_THRE,
                      label='Threshold value',
                      help='Gray values below this threshold are set to 0')
        form.addParam('volumeMask', PointerParam, pointerClass='VolumeMask',
                      label='Mask volume',
                      condition='maskMode==%d' % NMA_MASK_FILE,
                      )
        form.addParam('pseudoAtomRadius', FloatParam, default=1,
                      label='Pseudoatom radius (vox)',
                      help='Pseudoatoms are defined as Gaussians whose '
                           'standard deviation is this value in voxels.')
        form.addParam('pseudoAtomTarget', FloatParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Volume approximation error(%)',
                      help='This value is a percentage (between 0.001 and '
                           '100) specifying how fine you want to '
                           'approximate the EM volume by the pseudoatomic '
                           'structure. Lower values imply lower '
                           'approximation error, and consequently, more pseudoatoms.')

        # --------------------------- INSERT steps functions ----------------

    def _insertMaskStep(self, fnVol, prefix=''):
        """ Check the mask selected and insert the necessary steps.
        Return the mask filename if needed.
        """
        fnMask = ''
        if self.maskMode == NMA_MASK_THRE:
            fnMask = self._getExtraPath('mask%s.vol' % prefix)
            maskParams = '-i %s -o %s --select below %f --substitute binarize'\
                         % (fnVol, fnMask, self.maskThreshold.get())
            self.runJob("xmipp_transform_threshold", maskParams, env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        elif self.maskMode == NMA_MASK_FILE:
            fnMask = getImageLocation(self.volumeMask.get())
        return fnMask

    # --------------------------- STEPS functions ---------------------------
    def convertToPseudoAtomsStep(self, inputFn, fnMask, sampling, prefix=''):
        pseudoatoms = 'pseudoatoms%s' % prefix
        outputFn = self._getPath(pseudoatoms)
        sigma = sampling * self.pseudoAtomRadius.get()
        targetErr = self.pseudoAtomTarget.get()
        #volume-to-pseudoatom conversion was not MPI-parallelized and the number of MPIs was removed from the gui
        #nthreads = self.numberOfThreads.get() * self.numberOfMpi.get()
        nthreads = self.numberOfThreads.get()

        params = "-i %(inputFn)s -o %(outputFn)s --sigma %(sigma)f --thr " \
                 "%(nthreads)d "
        params += "--targetError %(targetErr)f --sampling_rate %(sampling)f " \
                  "-v 2 --intensityColumn Bfactor"
        if fnMask:
            params += " --mask binary_file %(fnMask)s"
        self.runJob("xmipp_volume_to_pseudoatoms", params % locals(), env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        for suffix in ["_approximation.vol", "_distance.hist"]:
            moveFile(self._getPath(pseudoatoms + suffix),
                     self._getExtraPath(pseudoatoms + suffix))
        self.runJob("xmipp_image_convert",
                    "-i %s_approximation.vol -o %s_approximation.mrc -t vol"
                    % (self._getExtraPath(pseudoatoms),
                       self._getExtraPath(pseudoatoms)), env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        self.runJob("xmipp_image_header",
                    "-i %s_approximation.mrc --sampling_rate %f" %
                    (self._getExtraPath(pseudoatoms), sampling),env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        cleanPattern(self._getPath(pseudoatoms + '_*'))

    def createChimeraScript(self, volume, pdb):
        """ Create a chimera script to visualize a pseudoatoms pdb
        obteined from a given EM 3d volume.
        A property will be set in the pdb object to
        store the location of the script.
        """
        pseudoatoms = pdb.getFileName()
        scriptFile = pseudoatoms + '_chimera.cxc'
        pdb._chimeraScript = String(scriptFile)
        sampling = volume.getSamplingRate()
        radius = sampling * self.pseudoAtomRadius.get()
        fnIn = getImageLocation(volume)
        if fnIn.endswith(":mrc"):
            fnIn = fnIn[:-4]

        x, y, z = volume.getOrigin(force=True).getShifts()
        xx, yy, zz = volume.getDim()

        dim = volume.getDim()[0]
        bildFileName = os.path.abspath(self._getExtraPath("axis.bild"))
        Chimera.createCoordinateAxisFile(dim,
                                 bildFileName=bildFileName,
                                 sampling=sampling)
        fhCxc = open(scriptFile, 'w')
        fhCxc.write("open %s\n" % basename(pseudoatoms))
        fhCxc.write("color by bfactor target a range 0,0.5\n")
        fhCxc.write("setattr a radius %f\n" % radius)
        fhCxc.write("style #1 sphere\n")
        modelID = 1
        fhCxc.write("open %s\n" % abspath(fnIn))
        threshold = 0.01
        if self.maskMode == NMA_MASK_THRE:
            self.maskThreshold.get()
        # set sampling
        fhCxc.write("volume #%d level %f transparency 0.5 voxelSize %f origin "
                    "%0.2f,%0.2f,%0.2f\n"
                    % (modelID + 1, threshold, sampling, x, y, z))
        fhCxc.write("open %s\n" % bildFileName)
        fhCxc.write("move %0.2f,%0.2f,%0.2f model #%d coord #%d\n"
                    % (x + (xx / 2. * sampling),
                       y + (yy / 2. * sampling),
                       z + (zz / 2. * sampling),
                       modelID, modelID + 2))
        fhCxc.close()
