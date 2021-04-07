# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# * IMPMC, UPMC Sorbonne University
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
# **************************************************************************
from pyworkflow.object import String
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pwem.convert import cifToPdb
from pyworkflow.utils.path import makePath, copyFile
from pyworkflow.protocol import params


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
DIMRED_MAPPINGS = [DIMRED_PCA, DIMRED_LLTSA, DIMRED_LPP, DIMRED_PPCA, DIMRED_NPE]


class FlexProtDimredPdb(ProtAnalysis3D):
    """ Protocol for applying dimentionality reduction on PDB files. """
    _label = 'pdb dimentionality reduction'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('pdbSource', EnumParam, default=0,
                      label='Source of PDBs',
                      choices=['After subtomogram synthesis', 'File pattern'],
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('pdbs', params.PointerParam, pointerClass='FlexProtSynthesizeSubtomo',
                      condition='pdbSource == 0',
                      label="Subtomogram synthesis",
                      help='All PDBs should have the same size')
        form.addParam('pdbs_file', params.PathParam,
                      condition='pdbSource == 1',
                      label="List of PDBs",
                      help='Use the file pattern as file location with /*.pdb')
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

        # form.addParallelSection(threads=0, mpi=8)

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        pdb_mat = self.getInputPdbs()
        reducedDim = self.reducedDim.get()
        method = self.dimredMethod.get()
        extraParams = self.extraParams.get('')
        deformationsFile = self.getDeformationFile()
        self._insertFunctionStep('performPDBdimred',
                                 pdb_mat,reducedDim,method,extraParams,deformationsFile)
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def performPDBdimred(self,pdb_mat,reducedDim,method,extraParams,deformationsFile):
        pdbs_list = [f for f in glob.glob(pdb_mat)]
        pdbs_list.sort()
        pdbs_matrix = []
        for pdbfn in pdbs_list:
            pdb_lines = self.readPDB(pdbfn)
            pdb_coordinates = np.array(self.PDB2List(pdb_lines))
            pdbs_matrix.append(np.reshape(pdb_coordinates, -1))
        deformationFile = self._getExtraPath('pdbs_mat.txt')
        # The deformationFile is for xmipp methods
        np.savetxt(deformationFile, pdbs_matrix, fmt="%s")

        rows, columns = np.shape(pdbs_matrix)
        outputMatrix = self.getOutputMatrixFile()
        methodName = DIMRED_VALUES[method]
        if methodName == 'None':
            copyFile(deformationsFile,outputMatrix)
            return

        if methodName == 'sklearn_PCA':
            # X = np.loadtxt(fname=deformationsFile)
            X = pdbs_matrix
            pca = decomposition.PCA(n_components=reducedDim)
            pca.fit(X)
            Y = pca.transform(X)
            np.savetxt(outputMatrix,Y)
            M = np.matmul(np.linalg.pinv(X),Y)
            mappingFile = self._getExtraPath('projector.txt')
            np.savetxt(mappingFile,M)
            # save the pca:
            pca_pickled = self._getExtraPath('pca_pickled.txt')
            dump(pca,pca_pickled)
        else:
            args = "-i %(deformationsFile)s -o %(outputMatrix)s -m %(methodName)s %(extraParams)s"
            args += "--din %(columns)d --samples %(rows)d --dout %(reducedDim)d"
            if method in DIMRED_MAPPINGS:
                mappingFile = self._getExtraPath('projector.txt')
                args += " --saveMapping %(mappingFile)s"
            self.runJob("xmipp_matrix_dimred", args % locals())


        print(pdb_mat)
        pass

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
        return ['harastani2020hybrid','Jin2014']

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

    def getInputPdbs(self):
        if self.pdbSource.get()==0:
            return self.pdbs.get()._getExtraPath('*.pdb')
        else:
            return self.pdbs_file.get()

    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def readPDB(self, fnIn):
        with open(fnIn) as f:
            lines = f.readlines()
        return lines

    def getDeformationFile(self):
        return self._getExtraPath('pdbs_mat.txt')

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
