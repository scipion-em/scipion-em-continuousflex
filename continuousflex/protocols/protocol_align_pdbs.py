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
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.protocol import params
from continuousflex.protocols.utilities.genesis_utilities import numpyArr2dcd, dcd2numpyArr
from .utilities.pdb_handler import ContinuousFlexPDBHandler
from pwem.objects import AtomStruct

import numpy as np
import glob
import pwem.emlib.metadata as md

PDB_SOURCE_PATTERN = 0
PDB_SOURCE_OBJECT = 1
PDB_SOURCE_TRAJECT = 2

class FlexProtAlignPdb(ProtAnalysis3D):
    """ Protocol to perform rigid body alignement on a set of PDB files. """
    _label = 'pdbs rigid body alignement'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('pdbSource', EnumParam, default=0,
                      label='Source of PDBs',
                      choices=['File pattern', 'Object', 'Trajectory Files'],
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('pdbs_file', params.PathParam,
                      condition='pdbSource == %i'%PDB_SOURCE_PATTERN,
                      label="List of PDBs",
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('setOfPDBs', params.PointerParam, pointerClass='SetOfPDBs, SetOfAtomStructs',
                      condition='pdbSource == %i'%PDB_SOURCE_OBJECT,
                      label="Set of PDBs",
                      help='Use a scipion object SetOfPDBs / SetOfAtomStructs')
        form.addParam('dcds_file', params.PathParam,
                      condition='pdbSource == %i'%PDB_SOURCE_TRAJECT,
                      label="DCD trajectory file (s)",
                      help='Use the file pattern as file location with /*.dcd')
        form.addParam('dcd_ref_pdb', params.PointerParam, pointerClass='AtomStruct',
                      condition='pdbSource == %i'%PDB_SOURCE_TRAJECT,
                      label="trajectory Reference PDB",
                      help='Reference PDB of the trajectory (Only used for structural information (Atom name, residue number etc)'
                           '. The coordinates inside this PDB are not used. The atoms number and position in the file must'
                           ' correspond to the DCD file. ')
        form.addParam('dcd_start', params.IntParam, default=0,
                      condition='pdbSource == %i'%PDB_SOURCE_TRAJECT,
                      label="Beginning of the trajectory",
                      help='Index of the desired begining of the trajectory', expertLevel=params.LEVEL_ADVANCED)
        form.addParam('dcd_end', params.IntParam, default=-1,
                      condition='pdbSource == %i'%PDB_SOURCE_TRAJECT,
                      label="Ending of the trajectory",
                      help='Index of the desired end of the trajectory', expertLevel=params.LEVEL_ADVANCED)
        form.addParam('dcd_step', params.IntParam, default=1,
                      condition='pdbSource == %i'%PDB_SOURCE_TRAJECT,
                      label="Step of the trajectory",
                      help='Step to skip points in the trajectory', expertLevel=params.LEVEL_ADVANCED)



        form.addParam('alignRefPDB', params.PointerParam, pointerClass='AtomStruct',
                      label="Alignement Reference PDB",
                      help='Reference PDB to align the PDBs with')
        form.addParam('matchingType', params.EnumParam, label="Match structures ?", default=0,
                      choices=['All structures are matching', 'Match chain name + res no',
                               'Match segment name + res no'],
                      help="Method to find atomic coordinates correspondence between the pdb set "
                           "coordinates and the reference PDB. The method will select the matching atoms"
                           " and sort them in the corresponding order. If the structures in the files are"
                           " already matching, choose All structures are matching")

        form.addParam('createOutput', params.BooleanParam, default=True,
                      label="Create output Set of PDBs ?",
                      help='Create output set. This step can be time consuming and not necessary if you are only '
                           ' interested by the alignement parameters. The aligned coordinate are conserved as DCD file '
                           'in the extra directory.'
                        , expertLevel=params.LEVEL_ADVANCED)

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readInputFiles')
        self._insertFunctionStep('rigidBodyAlignementStep')
        if self.createOutput.get():
            self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def readInputFiles(self):
        inputFiles = self.getInputFiles()

        # Get pdbs coordinates
        if self.pdbSource.get() == PDB_SOURCE_TRAJECT:
            pdbs_arr = dcd2numpyArr(inputFiles[0])
            start = self.dcd_start.get()
            stop = self.dcd_end.get() if self.dcd_end.get() != -1 else pdbs_arr.shape[0],
            step = self.dcd_step.get()
            pdbs_arr = pdbs_arr[start:stop:step]
            for i in range(1,len(inputFiles)):
                pdb_arr_i = dcd2numpyArr(inputFiles[i])[start:stop:step]
                pdbs_arr = np.concatenate((pdbs_arr, pdb_arr_i), axis=0)

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

    def rigidBodyAlignementStep(self):

        # open files
        inputPDB = ContinuousFlexPDBHandler(self.getPDBRef())
        refPDB = ContinuousFlexPDBHandler(self.alignRefPDB.get().getFileName())
        arrDCD = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ =arrDCD.shape
        alignXMD = md.MetaData()

        # find matching index between reference and pdbs
        if self.matchingType.get() == 1:
            idx_matching_atoms = inputPDB.matchPDBatoms(reference_pdb=refPDB, matchingType=0)
            refPDB.select_atoms(idx_matching_atoms[:, 1])
        elif self.matchingType.get() == 2:
            idx_matching_atoms = inputPDB.matchPDBatoms(reference_pdb=refPDB, matchingType=1)
            refPDB.select_atoms(idx_matching_atoms[:, 1])
        else:
            idx_matching_atoms = None
        refPDB.write_pdb(self._getExtraPath("reference.pdb"))

        # loop over all pdbs
        for i in range(nframe):
            print("Aligning PDB %i ... " %i)

            # rotate
            if self.matchingType.get() != 0 :
                coord = arrDCD[i][idx_matching_atoms[:, 0]]
            else:
                coord = arrDCD[i]
            rot_mat, tran = ContinuousFlexPDBHandler.alignCoords(refPDB.coords, coord)
            arrDCD[i] = (np.dot(arrDCD[i], rot_mat) + tran).astype(np.float32)

            # add to MD
            shftx, shfty, shftz = tran
            rot, tilt, psi, = matrix2eulerAngles(rot_mat)
            index = alignXMD.addObject()
            alignXMD.setValue(md.MDL_ANGLE_ROT, rot, index)
            alignXMD.setValue(md.MDL_ANGLE_TILT, tilt, index)
            alignXMD.setValue(md.MDL_ANGLE_PSI, psi, index)
            alignXMD.setValue(md.MDL_SHIFT_X, shftx, index)
            alignXMD.setValue(md.MDL_SHIFT_Y, shfty, index)
            alignXMD.setValue(md.MDL_SHIFT_Z, shftz, index)
            alignXMD.setValue(md.MDL_IMAGE, "", index)

        numpyArr2dcd(arrDCD, self._getExtraPath("coords.dcd"))
        alignXMD.write(self._getExtraPath("alignement.xmd"))


    def createOutputStep(self):
        pdbset = self._createSetOfPDBs("outputPDBs")
        arrDCD = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        refPDB = ContinuousFlexPDBHandler(self._getExtraPath("reference.pdb"))

        nframe, natom,_ = arrDCD.shape
        for i in range(nframe):
            filename = self._getExtraPath("output_%s.pdb" %str(i+1).zfill(6))
            refPDB.coords = arrDCD[i]
            refPDB.write_pdb(filename)
            pdb = AtomStruct(filename=filename)
            pdbset.append(pdb)

        self._defineOutputs(outputPDBs = pdbset)

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

    def getInputFiles(self):
        if self.pdbSource.get()==PDB_SOURCE_PATTERN:
            l= [f for f in glob.glob(self.pdbs_file.get())]
        elif self.pdbSource.get()==PDB_SOURCE_OBJECT:
            l= [i.getFileName() for i in self.setOfPDBs.get()]
        elif self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            l= [f for f in glob.glob(self.dcds_file.get())]
        l.sort()
        return l

    def getPDBRef(self):
        if self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            return self.dcd_ref_pdb.get().getFileName()
        else:
            return self.getInputFiles()[0]



def matrix2eulerAngles(A):
    abs_sb = np.sqrt(A[0, 2] * A[0, 2] + A[1, 2] * A[1, 2])
    if (abs_sb > 16 * np.exp(-5)):
        gamma = np.arctan2(A[1, 2], -A[0, 2])
        alpha = np.arctan2(A[2, 1], A[2, 0])
        if (abs(np.sin(gamma)) < np.exp(-5)):
            sign_sb = np.sign(-A[0, 2] / np.cos(gamma))
        else:
            if np.sin(gamma) > 0:
                sign_sb = np.sign(A[1, 2])
            else:
                sign_sb = -np.sign(A[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, A[2, 2])
    else:
        if (np.sign(A[2, 2]) > 0):
            alpha = 0
            beta = 0
            gamma = np.arctan2(-A[1, 0], A[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = np.arctan2(A[1, 0], -A[0, 0])
    gamma = np.rad2deg(gamma)
    beta = np.rad2deg(beta)
    alpha = np.rad2deg(alpha)
    return alpha, beta, gamma

