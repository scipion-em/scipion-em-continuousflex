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

import xmipp3.convert
from pyworkflow.protocol.params import PointerParam
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from subprocess import check_call
import sys
from os.path import isfile
import continuousflex
from pyworkflow.utils.path import copyFile
import pwem as em
import pwem.emlib.metadata as md
from xmipp3.convert import (createItemMatrix, setXmippAttributes)
from pyworkflow import BETA
from continuousflex import Plugin
from pwem.utils import runProgram

OPTION_NMA = 0
OPTION_ANGLES = 1
OPTION_SHFITS = 2
OPTION_ALL = 3

DEVICE_CUDA = 0
DEVICE_CPU = 1

class FlexProtDeepHEMNMAInfer(ProtAnalysis3D):
    """ This protocol is DeepHEMNMA
    """
    _label = 'deep hemnma infer'
    _devStatus = BETA

    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('analyze_option', params.EnumParam, label='set the parameter to predict',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['Predict Normal Mode Amplitudes',
                               'Predict Angles',
                               'Predict Shifts',
                               'predict All parameters',
                               ], default=OPTION_ALL,
                      help='select a set of parameter to predict')
        form.addParam('device_option', params.EnumParam, label='set the device for training',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['train on GPUs',
                               'tain on CPUs'], default=DEVICE_CUDA,
                      help='set a device to run the training on')
        form.addParam('trained_model', params.PointerParam, pointerClass='FlexProtDeepHEMNMATrain',
                      label = 'Trained model', help='import the training weights')
        form.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles',
                       label="Previous run of rigid-body alignment",
                       help='Select a previous run of rigid-body alignment.', allowsNull=True)
        form.addParam('num_modes', params.IntParam, label='Number of modes',default=3)
        form.addParam('batch_size', params.IntParam, expertLevel=params.LEVEL_ADVANCED, label='Batch size', default=2)
        form.addParallelSection(threads=0, mpi=0)    
    
    
    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('performDeepHEMNMAStep')
        self._insertFunctionStep('createOutputStep')
        
    #--------------------------- STEPS functions --------------------------------------------   
    
    def convertInputStep(self):
        xmipp3.convert.writeSetOfParticles(self.inputParticles.get(), self._getExtraPath('particles.xmd'))
        # copy atoms or pseudoatoms file from HEMNMA
        file = self.trained_model.get().inputNMA.get()._getExtraPath('atoms.pdb')
        if isfile(file):
            copyFile(file, self._getExtraPath('atoms.pdb'))
        else:
            copyFile(self.trained_model.get().inputNMA.get()._getExtraPath('pseudoatoms.pdb'), self._getExtraPath('pseudoatoms.pdb'))


    
    def performDeepHEMNMAStep(self):
        weights = self.trained_model.get()._getExtraPath('weights.pth')
        #copyFile(self.inputParticles.get('atoms.pdb'), self._getExtraPath('atoms.pdb'))
        #copyFile(self.inputParticles.get('modes.pdb'), self._getExtraPath('modes.pdb'))
        batch_size = self.batch_size.get()
        mode = self.analyze_option.get()
        device = self.device_option.get()
        num_modes = self.num_modes.get()
        self.imgsFn = self._getExtraPath('particles.xmd')
        params = " %s %s %s %d %d %d %d" % (self.imgsFn, weights, self._getExtraPath(), num_modes, batch_size, mode, device)
        script_path = continuousflex.__path__[0]+'/protocols/utilities/deep_hemnma_infer.py'
        command = "python " + script_path + params
        command = Plugin.getContinuousFlexCmd(command)
        check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr, env=None, cwd=None)
        pass

        
    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(inputSet)
        self.imgsFn = self._getExtraPath('images.xmd')
        copyFile(self.imgsFn, self._getExtraPath('infer.xmd'))
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self.imgsFn, sortByLabel=md.MDL_ITEM_ID))
        partSet.copyItems(self.trained_model.get().inputNMA.get().outputParticles)
        self._defineOutputs(outputParticles=partSet)
        # Lets write a metadata that combines both of these training and inference sets:
        fn_train = self.trained_model.get().inputNMA.get()._getExtraPath('images.xmd')
        fn_infer = self._getExtraPath('infer.xmd')
        fn_combined = self._getExtraPath('images.xmd')
        args = '-i %(fn_train)s -o %(fn_combined)s --set union %(fn_infer)s' % locals()
        runProgram('xmipp_metadata_utilities', args)


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


    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
                           md.MDL_SHIFT_Y, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)