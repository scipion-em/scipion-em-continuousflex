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
    """
    This protocol is DeepHEMNMA

    AI Generated:

    Deep HEMNMA Inference (FlexProtDeepHEMNMAInfer) - User Manual
        Overview

        The Deep HEMNMA Inference protocol applies a previously trained Deep HEMNMA model to
        estimate structural variability parameters directly from cryo-EM particle images. Its
        primary objective is to use knowledge learned during a prior training stage to rapidly
        predict conformational and rigid-body descriptors for new datasets, avoiding the need to
        perform a complete variability analysis from the beginning.

        In studies of molecular flexibility, researchers often need to characterize large numbers
        of particles that may represent a continuum of structural states. Once a predictive model
        has been trained, this protocol enables efficient estimation of the underlying variability
        parameters, making large-scale analyses substantially more practical.

        Inputs and General Workflow

        The protocol requires a trained Deep HEMNMA model together with a new particle dataset that
        will be analyzed. The trained model serves as a learned representation of the relationship
        between particle appearance and structural variability, while the input particles provide
        the experimental observations from which predictions are generated.

        The protocol is intended to be used after a successful training stage. The quality of the
        predictions depends strongly on how well the training dataset represents the structural
        variability present in the new particles. Datasets that differ substantially from the data
        used during training may produce less reliable results.

        Prediction Targets

        Different categories of structural descriptors can be predicted depending on the scientific
        objective. Conformational variability can be represented through normal mode amplitudes,
        which describe collective molecular motions and continuous structural transitions between
        biological states.

        The protocol can also estimate rotational and translational parameters associated with
        rigid-body variability. These descriptors are useful when structural differences are
        dominated by large-scale movements of domains, subunits, or complete molecular assemblies.

        In many applications, predicting all available parameters simultaneously provides the most
        comprehensive description of particle heterogeneity. However, restricting the prediction
        task to a specific category may be advantageous when the biological question focuses on a
        particular type of motion.

        Normal Modes and Structural Interpretation

        The number of normal modes determines the dimensionality of the conformational description.
        A larger number of modes can capture more complex motions, while a smaller number often
        focuses on the dominant collective movements that explain most of the observed variability.

        From a biological perspective, normal mode amplitudes should be interpreted as coordinates
        within a continuous conformational landscape. Similar amplitude values generally correspond
        to related structural states, whereas larger differences may indicate distinct conformations
        or transitions between functional forms.

        Computational Resources

        The protocol supports execution on graphical processing units and conventional central
        processing units. GPU execution is generally preferred because neural network inference can
        be performed significantly faster, especially when processing large particle collections.

        CPU execution remains useful when accelerator hardware is unavailable or when analyzing
        smaller datasets. The choice of computational device affects performance but does not
        change the biological interpretation of the predicted parameters.

        Outputs and Their Interpretation

        The protocol produces a particle set enriched with predicted structural descriptors. These
        predictions can be used for downstream analyses of conformational variability, structural
        clustering, visualization of continuous motions, or comparison with previously characterized
        states.

        The resulting dataset preserves the connection between each particle and its estimated
        variability parameters, enabling researchers to explore structural landscapes at the
        individual-particle level. This information can be valuable for identifying dominant motions,
        mapping functional transitions, or studying heterogeneous molecular assemblies.

        In addition to generating predictions for the new dataset, the protocol integrates the
        inferred information with the variability information associated with the training data.
        This facilitates direct comparison between previously characterized particles and newly
        analyzed observations within a common variability framework.

        Practical Recommendations

        For best results, the training model should originate from a dataset that adequately samples
        the conformational space expected in the inference dataset. Predictions are generally more
        reliable when the new particles belong to the same biological system and imaging conditions
        used during training.

        Researchers should carefully select the prediction target according to their biological
        objectives. Studies focused on molecular flexibility often benefit from emphasizing normal
        mode amplitudes, whereas investigations involving particle orientation or positional
        variability may require angular and translational predictions as well.

        When working with very large datasets, GPU execution is typically the most efficient option.
        It allows rapid processing while maintaining the same predictive framework established during
        training.

        Final Perspective

        Deep HEMNMA Inference transforms a trained deep learning model into a practical tool for
        exploring structural heterogeneity in cryo-EM data. By predicting conformational and
        rigid-body descriptors directly from particle images, it enables efficient characterization
        of molecular variability and supports the study of continuous structural landscapes across
        large experimental datasets.
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