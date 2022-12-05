# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
# *             Slavica Jonic                (slavica.jonic@upmc.fr)
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

from pyworkflow.object import String
from pyworkflow.protocol.params import (PointerParam, StringParam, EnumParam, IntParam,
                                        LEVEL_ADVANCED)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils.path import copyFile
import numpy as np
from sklearn import decomposition
from joblib import dump
import xmipp3

DIMRED_PCA = 0
DIMRED_LTSA = 1
DIMRED_DM = 2
DIMRED_LLTSA = 3
DIMRED_LPP = 4
DIMRED_KPCA = 5
DIMRED_PPCA = 6
DIMRED_LE = 7
DIMRED_HLLE = 8
DIMRED_SPE = 9
DIMRED_NPE = 10
DIMRED_SKLEAN_PCA = 11

# Values to be passed to the program
DIMRED_VALUES = ['PCA', 'LTSA', 'DM', 'LLTSA', 'LPP', 'kPCA', 'pPCA', 'LE', 'HLLE', 'SPE', 'NPE', 'sklearn_PCA','None']

# Methods that allows mapping
DIMRED_MAPPINGS = [DIMRED_PCA, DIMRED_LLTSA, DIMRED_LPP, DIMRED_PPCA, DIMRED_NPE]


class FlexProtDimredHeteroFlow(ProtAnalysis3D):
    """ This protocol will take volumes with optical flows, it will operate on the correlation mat
    and will project it onto a reduced space
    """
    _label = 'tomoflow dimred'

    def __init__(self, **kwargs):
        ProtAnalysis3D.__init__(self, **kwargs)
        self.mappingFile = String()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputOpFlow', PointerParam, pointerClass='FlexProtHeteroFlow',
                      label="Optical flows",
                      help='Select a previous run of optical flow for volumes and a reference.')

        form.addParam('dimredMethod', EnumParam, default=DIMRED_SKLEAN_PCA,
                      choices=['Principal Component Analysis (PCA)',
                               'Local Tangent Space Alignment',
                               'Diffusion map',
                               'Linear Local Tangent Space Alignment',
                               'Linearity Preserving Projection',
                               'Kernel PCA',
                               'Probabilistic PCA',
                               'Laplacian Eigenmap',
                               'Hessian Locally Linear Embedding',
                               'Stochastic Proximity Embedding',
                               'Neighborhood Preserving Embedding',
                               'Scikit-Learn PCA',
                               "Don't reduce dimensions"],
                      label='Dimensionality reduction method',
                      help=""" Choose among the following dimensionality reduction methods:
    PCA
       Principal Component Analysis 
    LTSA <k=12>
       Local Tangent Space Alignment, k=number of nearest neighbours 
    DM <s=1> <t=1>
       Diffusion map, t=Markov random walk, s=kernel sigma 
    LLTSA <k=12>
       Linear Local Tangent Space Alignment, k=number of nearest neighbours 
    LPP <k=12> <s=1>
       Linearity Preserving Projection, k=number of nearest neighbours, s=kernel sigma 
    kPCA <s=1>
       Kernel PCA, s=kernel sigma 
    pPCA <n=200>
       Probabilistic PCA, n=number of iterations 
    LE <k=7> <s=1>
       Laplacian Eigenmap, k=number of nearest neighbours, s=kernel sigma 
    HLLE <k=12>
       Hessian Locally Linear Embedding, k=number of nearest neighbours 
    SPE <k=12> <global=1>
       Stochastic Proximity Embedding, k=number of nearest neighbours, global embedding or not 
    NPE <k=12>
       Neighborhood Preserving Embedding, k=number of nearest neighbours 
""")
        form.addParam('extraParams', StringParam,
                      expertLevel=LEVEL_ADVANCED,
                      label="Extra params",
                      help='These parameters are there to change the default parameters of a dimensionality reduction'
                           ' method. Check xmipp_matrix_dimred for full details.')

        form.addParam('reducedDim', IntParam, default=2,
                      label='Reduced dimension')
        form.addParallelSection(threads=0, mpi=0)

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Take deforamtions text file and the number of images and modes
        inputSet = self.getInputParticles()
        rows = inputSet.getSize()
        # rows = inputSet.get().getSize()
        reducedDim = self.reducedDim.get()
        method = self.dimredMethod.get()
        extraParams = self.extraParams.get('')
        deformationsFile = self.getDeformationFile()

        self._insertFunctionStep('convertInputStep',
                                 deformationsFile)
        self._insertFunctionStep('performDimredStep',
                                 deformationsFile, method, extraParams,
                                 rows, reducedDim)
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, deformationFile):
        """ Copy the data.csv file that will serve as
        input for dimensionality reduction.
        """
        inputSet = self.getInputParticles()
        # copy the reference abd the deformations file
        reference = self.inputOpFlow.get()._getExtraPath('reference.spi')
        copyFile(reference,self._getExtraPath('reference.spi'))
        data = self.inputOpFlow.get()._getExtraPath('data.csv')
        copyFile(data,deformationFile)


    def performDimredStep(self, deformationsFile, method, extraParams,
                          rows, reducedDim):
        outputMatrix = self.getOutputMatrixFile()
        methodName = DIMRED_VALUES[method]
        if methodName == 'None':
            copyFile(deformationsFile,outputMatrix)
            return
        # Get number of columes in deformation files
        # it can be a subset of inputModes

        # convert the file from comma separated to spcae separated for compitability
        data = np.loadtxt(deformationsFile, delimiter=',')
        np.savetxt(deformationsFile, data, delimiter=' ')

        f = open(deformationsFile)
        columns = len(f.readline().split())  # count number of values in first line
        f.close()

        if methodName == 'sklearn_PCA':
            X = np.loadtxt(fname=deformationsFile)
            pca = decomposition.PCA(n_components=reducedDim)
            pca.fit(X)
            Y = pca.transform(X)
            np.savetxt(outputMatrix,Y)
            M = np.matmul(np.linalg.pinv(X),Y)
            mappingFile = self._getExtraPath('projector.txt')
            np.savetxt(mappingFile,M)
            self.mappingFile.set(mappingFile)
            # save the pca:
            pca_pickled = self._getExtraPath('pca_pickled.txt')
            dump(pca,pca_pickled)

        else:
            args = "-i %(deformationsFile)s -o %(outputMatrix)s -m %(methodName)s %(extraParams)s"
            args += "--din %(columns)d --samples %(rows)d --dout %(reducedDim)d"
            if method in DIMRED_MAPPINGS:
                mappingFile = self._getExtraPath('projector.txt')
                args += " --saveMapping %(mappingFile)s"
                self.mappingFile.set(mappingFile)
            self.runJob("xmipp_matrix_dimred", args % locals())

    def createOutputStep(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------

    def getInputParticles(self):
        """ Get the particles of the input optical flow protocol. """
        if(self.inputOpFlow.get().inputVolumes.get()):
            return self.inputOpFlow.get().inputVolumes.get()
        else:
            # number of refinement iterations
            num = self.inputOpFlow.get().refinementProt.get().NumOfIters.get()+1
            fn = 'volumes_aligned_'+str(num)+'.xmd'
            mdfn = self.inputOpFlow.get().refinementProt.get()._getExtraPath(fn)
            partSet = self._createSetOfVolumes('to_average')
            xmipp3.convert.readSetOfVolumes(mdfn, partSet)
            partSet.setSamplingRate(self.inputOpFlow.get().refinementProt.get().inputVolumes.get().getSamplingRate())
            return partSet

    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def getDeformationFile(self):
        return self._getExtraPath('deformations.txt')

    def getProjectorFile(self):
        return self.mappingFile.get()

    def getMethodName(self):
        return DIMRED_VALUES[self.dimredMethod.get()]

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022tomoflow','harastani2022continuousflex']

    def _methods(self):
        return []
