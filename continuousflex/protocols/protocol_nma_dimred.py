# **************************************************************************
# *
# * Authors:
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Nov 2014
# * Slavica Jonic (slavica.jonic@upmc.fr)
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
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
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pwem.convert import cifToPdb
from pyworkflow.utils.path import makePath, copyFile
from pyworkflow.protocol import params
from pwem.utils import runProgram
from continuousflex.protocols import FlexProtAlignmentNMA

import numpy as np
import glob
from sklearn import decomposition
from joblib import dump

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

USE_PDBS = 0
USE_NMA_AMP = 1

# Values to be passed to the program
DIMRED_VALUES = ['PCA', 'LTSA', 'DM', 'LLTSA', 'LPP', 'kPCA', 'pPCA', 'LE', 'HLLE', 'SPE', 'NPE', 'sklearn_PCA','None']

# Methods that allows mapping
DIMRED_MAPPINGS = [DIMRED_PCA, DIMRED_LLTSA, DIMRED_LPP, DIMRED_PPCA, DIMRED_NPE]

DATA_CHOICE = ['PDBs', 'NMAs']


class FlexProtDimredNMA(ProtAnalysis3D):
    """ This protocol will take the volumes with NMA deformations
    as points in a N-dimensional space (where N is the number
    of computed normal modes) and will project them onto a reduced space
    """
    _label = 'nma dimred'
    
    def __init__(self, **kwargs):
        ProtAnalysis3D.__init__(self, **kwargs)
        self.mappingFile = String()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputNMA', PointerParam, pointerClass='FlexProtAlignmentNMA, FlexProtDeepHEMNMAInfer',
                      label="Conformational distribution",
                      help='Select a previous run of the NMA alignment.')

        form.addParam('dataChoice', EnumParam, default=USE_NMA_AMP,
                      choices=['Use deformed (pseudo)atomic models',
                               'Use normal mode amplitudes'],
                      label='Data to analyze',
                      help='Theoretically, both methods should give similar results, but choosing to analyze the fitted'
                           ' PDBs can help reduce / eliminate the crosstalk between the normal-modes.'
                           ' We recommend trying both options and comparing the results.')

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
        form.addParam('extraParams', params.StringParam, default=None,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Extra params',
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
        reducedDim = self.reducedDim.get()
        method = self.dimredMethod.get()
        extraParams = self.extraParams.get('')
        dataChoice = self.getDataChoice()
        deformationsFile = self.getDeformationFile()

        self._insertFunctionStep('convertInputStep',
                                 deformationsFile, inputSet.getObjId(), dataChoice)
        self._insertFunctionStep('performDimredStep',
                                 deformationsFile, method, extraParams,
                                 rows, reducedDim)
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, deformationFile, inputId, dataChoice):
        """ Iterate through the volumes and write the
        plain deformation.txt file that will serve as
        input for dimensionality reduction.
        """
        inputSet = self.getInputParticles()

        if dataChoice == 'NMAs':
            f = open(deformationFile, 'w')
            for particle in inputSet:
                f.write(' '.join(particle._xmipp_nmaDisplacements))
                f.write('\n')
            f.close()
        elif dataChoice == 'PDBs':
            # copy the pdb
            input_pdbfn = self.getInputPdb().getFileName()
            pdbfn = self._getExtraPath('pdb_file.pdb')
            self.copyinputPdb(input_pdbfn, pdbfn)
            # use the deformations to generate deformed versions of the pdb:
            selected_nma_modes = self.getInputModes()
            nma_amplfn = self._getExtraPath('nma_amplitudes.txt')
            f = open(nma_amplfn, 'w')
            for particle in inputSet:
                f.write(' '.join(particle._xmipp_nmaDisplacements))
                f.write('\n')
            f.close()
            nma_ampl = np.loadtxt(nma_amplfn)
            makePath(self._getExtraPath('generated_pdbs'))
            pdbs_folder = self._getExtraPath('generated_pdbs')
            i = 1
            for line in nma_ampl:
                cmd = '-o ' + pdbs_folder + '/' + str(i).zfill(
                    6) + '.pdb' + ' --pdb ' + pdbfn + ' --nma ' + selected_nma_modes + \
                      ' --deformations ' + ' '.join(map(str, line))
                #print(cmd)
                runProgram('xmipp_pdb_nma_deform', cmd)
                i += 1
            pdbs_list = [f for f in glob.glob(pdbs_folder+'/*.pdb')]
            pdbs_list.sort()
            pdbs_matrix = []
            for pdbfn in pdbs_list:
                pdb_lines = self.readPDB(pdbfn)
                pdb_coordinates = np.array(self.PDB2List(pdb_lines))
                pdbs_matrix.append(np.reshape(pdb_coordinates, -1))
            np.savetxt(deformationFile, pdbs_matrix, fmt="%s")
            pass

        else:
            print('Data for dimensionality reduction is not set correctly')

    def performDimredStep(self, deformationsFile, method, extraParams,
                          rows, reducedDim):
        outputMatrix = self.getOutputMatrixFile()
        methodName = DIMRED_VALUES[method]
        if methodName == 'None':
            copyFile(deformationsFile,outputMatrix)
            return
        # Get number of columes in deformation files
        # it can be a subset of inputModes
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
            runProgram("xmipp_matrix_dimred", args % locals())

    def createOutputStep(self):
        pass

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022continuousflex','Jin2014']

    def _methods(self):
        return []

    # --------------------------- UTILS functions --------------------------------------------
    def getInputModes(self):
        if isinstance(self.inputNMA.get(), FlexProtAlignmentNMA):
            return self.inputNMA.get()._getExtraPath('modes.xmd')
        else:
            return self.inputNMA.get().trained_model.get().inputNMA.get()._getExtraPath('modes.xmd')

    def getInputParticles(self):
        """ Get the output particles of the input NMA protocol. """
        return self.inputNMA.get().outputParticles

    def getParticlesMD(self):
        "Get the metadata files that contain the NMA displacement"
        return self.inputNMA.get()._getExtraPath('images.xmd')

    def getInputPdb(self):
        if isinstance(self.inputNMA.get(), FlexProtAlignmentNMA):
            return self.inputNMA.get().getInputPdb()
        else:
            return self.inputNMA.get().trained_model.get().inputNMA.get().getInputPdb()


    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def getDeformationFile(self):
        return self._getExtraPath('deformations.txt')

    def getProjectorFile(self):
        return self.mappingFile.get()

    def getMethodName(self):
        return DIMRED_VALUES[self.dimredMethod.get()]

    def getDataChoice(self):
        return DATA_CHOICE[self.dataChoice.get()]

    def copyinputPdb(self, inputFn, localFn):
        """ Copy the input pdb file
        """
        # if it is not cif, no problem, it will keep a pdb as it is and copy it
        cifToPdb(inputFn, localFn)

    def readPDB(self, fnIn):
        with open(fnIn) as f:
            lines = f.readlines()
        return lines

    def PDB2List(self, lines):
        newlines = []
        for line in lines:
            if line.startswith("ATOM "):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    newline = [x, y, z]
                    newlines.append(newline)
                except:
                    pass
        return newlines
