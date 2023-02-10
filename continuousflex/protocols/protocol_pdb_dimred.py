# **************************************************************************
# * Author:  Remi Vuillemot
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

import joblib
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils.path import makePath
from pyworkflow.protocol import params
from pwem.emlib import MetaData, MDL_ENABLED, MDL_NMA_MODEFILE,MDL_ORDER
from pwem.objects import SetOfNormalModes, AtomStruct
from .convert import rowToMode
from xmipp3.base import XmippMdRow
from continuousflex.protocols.utilities.genesis_utilities import numpyArr2dcd,dcd2numpyArr

import numpy as np
import glob
from sklearn import decomposition
from joblib import dump

from .utilities.pdb_handler import ContinuousFlexPDBHandler
import continuousflex
from continuousflex import Plugin
from subprocess import check_call
import sys

PDB_SOURCE_SUBTOMO = 0
PDB_SOURCE_PATTERN = 1
PDB_SOURCE_OBJECT = 2
PDB_SOURCE_TRAJECT = 3
PDB_SOURCE_ALIGNED = 4

REDUCE_METHOD_PCA = 0
REDUCE_METHOD_UMAP = 1


class FlexProtDimredPdb(ProtAnalysis3D):
    """ Protocol for applying dimentionality reduction on PDB files. """
    _label = 'pdb dimentionality reduction'


    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('pdbSource', EnumParam, default=PDB_SOURCE_SUBTOMO,
                      label='Source of PDBs',
                      choices=['Used for subtomogram synthesis', 'File pattern', 'Object', 'Trajectory Files', 'Align PDBs protocol'],
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('pdbs', params.PointerParam, pointerClass='FlexProtSynthesizeSubtomo',
                      condition='pdbSource ==  %i'%PDB_SOURCE_SUBTOMO,
                      label="Subtomogram synthesis",
                      help='Point to a protocol of synthesizing subtomograms, the ground truth PDBs will be used as input')
        form.addParam('pdbs_file', params.PathParam,
                      condition='pdbSource == %i' % PDB_SOURCE_PATTERN,
                      label="List of PDBs",
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('setOfPDBs', params.PointerParam, pointerClass='SetOfPDBs, SetOfAtomStructs',
                      condition='pdbSource == %i' % PDB_SOURCE_OBJECT,
                      label="Set of PDBs",
                      help='Use a scipion object SetOfPDBs / SetOfAtomStructs')
        form.addParam('dcds_file', params.PathParam,
                      condition='pdbSource == %i' % PDB_SOURCE_TRAJECT,
                      label="DCD trajectory file (s)",
                      help='Use the file pattern as file location with /*.dcd')
        form.addParam('dcd_ref_pdb', params.PointerParam, pointerClass='AtomStruct',
                      condition='pdbSource == %i' % PDB_SOURCE_TRAJECT,
                      label="trajectory Reference PDB",
                      help='Reference PDB of the trajectory (Only used for structural information (Atom name, residue number etc)'
                           '. The coordinates inside this PDB are not used. The atoms number and position in the file must'
                           ' correspond to the DCD file. ')
        form.addParam('dcd_start', params.IntParam, default=0,
                      condition='pdbSource == %i' % PDB_SOURCE_TRAJECT,
                      label="Beginning of the trajectory",
                      help='Index of the desired begining of the trajectory', expertLevel=params.LEVEL_ADVANCED)
        form.addParam('dcd_end', params.IntParam, default=-1,
                      condition='pdbSource == %i' % PDB_SOURCE_TRAJECT,
                      label="Ending of the trajectory",
                      help='Index of the desired end of the trajectory', expertLevel=params.LEVEL_ADVANCED)
        form.addParam('dcd_step', params.IntParam, default=1,
                      condition='pdbSource == %i' % PDB_SOURCE_TRAJECT,
                      label="Step of the trajectory",
                      help='Step to skip points in the trajectory', expertLevel=params.LEVEL_ADVANCED)

        form.addParam('alignPdbProt', params.PointerParam, pointerClass='FlexProtAlignPdb',
                      condition='pdbSource == %i' % PDB_SOURCE_ALIGNED,
                      label="Align PDBs Protocol",
                      help='Point to a protocol of pdb aligned. For large data set, you can use here the align pdb protocol as input '
                           'and avoid creating an output set of pdb in the align pdb protocol.')

        form.addParam('method', params.EnumParam, label="Reduction method", default=REDUCE_METHOD_PCA,
                      choices=['PCA', 'UMAP'],help="")

        form.addParam('reducedDim', IntParam, default=10,
                      label='Number of Principal Components')

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readInputFiles')
        self._insertFunctionStep('performDimred')
        if self.method.get() == REDUCE_METHOD_PCA:
            self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def readInputFiles(self):
        inputFiles = self.getInputFiles()

        # Get pdbs coordinates
        if self.pdbSource.get() == PDB_SOURCE_TRAJECT:
            pdbs_arr = dcd2numpyArr(inputFiles[0])
            start = self.dcd_start.get()
            step = self.dcd_step.get()
            end = self.dcd_end.get() if self.dcd_end.get() != -1 else pdbs_arr.shape[0]
            pdbs_arr = pdbs_arr[start:end:step]
            for i in range(1,len(inputFiles)):
                pdb_arr_i = dcd2numpyArr(inputFiles[i])
                pdbs_arr = np.concatenate((pdbs_arr, pdb_arr_i[start:end:step]), axis=0)


        elif self.pdbSource.get() == PDB_SOURCE_ALIGNED:
            pdbs_arr = dcd2numpyArr(inputFiles[0])
        else:
            pdbs_matrix = []
            for pdbfn in inputFiles:
                try:
                    # Read PDBs
                    mol = ContinuousFlexPDBHandler(pdbfn)
                    pdbs_matrix.append(mol.coords)
                except RuntimeError:
                    print("Warning : Can not read PDB file %s " % pdbfn)
            pdbs_arr = np.array(pdbs_matrix)

        # save as dcd file
        numpyArr2dcd(pdbs_arr, self._getExtraPath("coords.dcd"))

    def performDimred(self):

        pdbs_arr = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ = pdbs_arr.shape
        pdbs_matrix = pdbs_arr.reshape(nframe, natom*3)

        if self.method.get() == REDUCE_METHOD_PCA:
            pca = decomposition.PCA(n_components=self.reducedDim.get())
            Y = pca.fit_transform(pdbs_matrix)
            dump(pca, self._getExtraPath('pca_pickled.joblib'))

            pathPC = self._getPath("modes")
            pdb = ContinuousFlexPDBHandler(self.getPDBRef())
            pdb.coords = pca.mean_.reshape(pdbs_matrix.shape[1] // 3, 3)
            pdb.write_pdb(self._getPath("atoms.pdb"))
            makePath(pathPC)
            matrix = pca.components_.reshape(self.reducedDim.get(),pdbs_matrix.shape[1]//3,3)
            self.writePrincipalComponents(prefix=pathPC, matrix = matrix)

        elif self.method.get() == REDUCE_METHOD_UMAP:
            pdbs_dump = self._getTmpPath('pdbs_dump.pkl')
            joblib.dump(pdbs_matrix, pdbs_dump)
            Y_dump = self._getTmpPath('Y_dump.pkl')
            args = "%d %d %d %s %s %s" % (self.reducedDim.get(), 15, 1000,
                                       pdbs_dump, self._getExtraPath('pca_pickled.joblib'), Y_dump)
            script_path = continuousflex.__path__[0] + '/protocols/utilities/umap_run.py '
            command = "python " + script_path + args
            command = Plugin.getContinuousFlexCmd(command)
            check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                       env=None, cwd=None)
            Y = joblib.load(Y_dump)

        np.savetxt(self.getOutputMatrixFile(),Y)

    def createOutputStep(self):
            # Metadata
            mdOut = MetaData()
            for i in range(self.reducedDim.get()):
                objId = mdOut.addObject()
                modefile = self._getPath("modes", "vec.%d" % (i + 1))
                mdOut.setValue(MDL_NMA_MODEFILE, modefile, objId)
                mdOut.setValue(MDL_ORDER, i + 1, objId)
                mdOut.setValue(MDL_ENABLED, 1, objId)
            mdOut.write(self._getPath("modes.xmd"))

            # Sqlite object
            pcSet =SetOfNormalModes(filename=self._getPath("modes.sqlite"))
            row = XmippMdRow()
            for objId in mdOut:
                row.readFromMd(mdOut, objId)
                pcSet.append(rowToMode(row))

            pdb = AtomStruct(self._getPath("atoms.pdb"))
            self._defineOutputs(outputMean=pdb)

            pcSet.setPdb(pdb)
            self._defineOutputs(outputPCA=pcSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

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

    def getInputFiles(self):
        if self.pdbSource.get()==PDB_SOURCE_SUBTOMO:
            l= [f for f in glob.glob(self.pdbs.get()._getExtraPath('*.pdb'))]
        elif self.pdbSource.get()==PDB_SOURCE_PATTERN:
            l= [f for f in glob.glob(self.pdbs_file.get())]
        elif self.pdbSource.get()==PDB_SOURCE_OBJECT:
            l= [i.getFileName() for i in self.setOfPDBs.get()]
        elif self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            l= [f for f in glob.glob(self.dcds_file.get())]
        elif self.pdbSource.get()==PDB_SOURCE_ALIGNED:
            l=[self.alignPdbProt.get()._getExtraPath("coords.dcd")]
        l.sort()
        return l

    def getPDBRef(self):
        if self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            return self.dcd_ref_pdb.get().getFileName()
        elif self.pdbSource.get()==PDB_SOURCE_ALIGNED:
            return self.alignPdbProt.get()._getExtraPath("reference.pdb")
        else:
            return self.getInputFiles()[0]

    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def writePrincipalComponents(self, prefix, matrix):
        for i in range(self.reducedDim.get()):
            with open("%s/vec.%i"%(prefix,i+1), "w") as f:
                    for j in range(matrix.shape[1]):
                        f.write(" %e   %e   %e\n" % (matrix[i,j, 0], matrix[i,j, 1], matrix[i,j, 2]))
