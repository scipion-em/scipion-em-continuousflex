# **************************************************************************
# *
# * Authors:
# * Ilyes Hamitouche
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

from pyworkflow.protocol.params import PointerParam
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from subprocess import check_call
import sys
import continuousflex
from pyworkflow import BETA
from continuousflex import Plugin

OPTION_NMA = 0
OPTION_ANGLES = 1
OPTION_SHFITS = 2
OPTION_ALL = 3

DEVICE_CUDA = 0
DEVICE_CPU = 1


class FlexProtDeepHEMNMATrain(ProtAnalysis3D):
    """ DeepHEMNMA protocol, a neural network that learns the rigid-body parameters and the normal mode
        amplitudes estimated by HEMNMA protocol.
    """
    _label = 'deep hemnma train'
    _devStatus = BETA

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('analyze_option', params.EnumParam, label='set the parameter to train on',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['Train on Normal Mode Amplitudes',
                               'Train on Angles',
                               'Train on Shifts',
                               'Train on All parameters',
                               ], default = OPTION_ALL,
                      help='select a set of parameter to train on')
        group = form.addGroup('Train on conformational variability', condition='analyze_option == %d or analyze_option == %d'% (OPTION_NMA, OPTION_ALL))
        group.addParam('inputNMA', PointerParam, pointerClass='FlexProtAlignmentNMA',
                      label="Previous HEMNMA run",
                      help='Select a previous run of the NMA image alignment.', allowsNull=True)
        group = form.addGroup('Train on rigid-body variability ', condition='analyze_option == %d or analyze_option == %d' %(OPTION_SHFITS, OPTION_ANGLES))
        group.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles',
                      label="Preious run of rigid-body alignment",
                      help='Select a previous run of rigid-body alignment.', allowsNull=True)
        form.addParam('device_option', params.EnumParam, label='set the device for training',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['train on GPUs',
                               'tain on CPUs'], default = DEVICE_CUDA,
                      help='set a device to run the training on')
        form.addParam('learning_rate', params.FloatParam, label = 'Learning rate', default = 0.0001)
        form.addParam('epochs', params.IntParam, expertLevel=params.LEVEL_ADVANCED,label = 'Number of epochs', default = 400)
        form.addParam('batch_size', params.IntParam ,expertLevel=params.LEVEL_ADVANCED, label = 'Batch size', default = 2)
        form.addParallelSection(threads=0, mpi=0)

    
    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('performDeepHEMNMAStep')
        self._insertFunctionStep('createOutputStep')
        
    #--------------------------- STEPS functions --------------------------------------------

    def performDeepHEMNMAStep(self):
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()
        lr = self.learning_rate.get()
        mode = self.analyze_option.get()
        device = self.device_option.get()
        imgsFn = self.inputNMA.get()._getExtraPath('images.xmd')

        params = " %s %s %d %d %f %d %d" % (imgsFn, self._getExtraPath(), epochs, batch_size, lr, mode, device)
        script_path = continuousflex.__path__[0]+'/protocols/utilities/deep_hemnma.py'
        command = "python " + script_path + params
        command = Plugin.getContinuousFlexCmd(command)
        check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr, env=None, cwd=None)


    def createOutputStep(self):
        pass

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary
    
    def _validate(self):
        errors = []
        return errors
    
    def _citations(self):
        return ['harastani2022continuousflex','hamitouche2022deephemnma']
    
    def _methods(self):
        return []
    
    #--------------------------- UTILS functions --------------------------------------------

    def getInputParticles(self):
        """ Get the output particles of the input NMA protocol. """
        return self.inputNMA.get().outputParticles

    def getParticlesMD(self):
        "Get the metadata files that contain the NMA displacement"
        return self.inputNMA.get()._getExtraPath('images.xmd')

    def getInputPdb(self):
        return self.inputNMA.get().getInputPdb()
    
    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')
    
    def getDeformationFile(self):
        return self._getExtraPath('deformations.txt')

    def getProjectorFile(self):
        return self.mappingFile.get()

