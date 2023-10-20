# **************************************************************************
# * Authors:  Remi Vuillemot             (remi.vuillemot@upmc.fr)
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
from pwem.objects import AtomStruct, SetOfParticles, SetOfVolumes
from continuousflex.protocols.convert import matrix2eulerAngles

import numpy as np

PDB_SOURCE_PATTERN = 0
PDB_SOURCE_OBJECT = 1
PDB_SOURCE_TRAJECT = 2
class FlexProtCA2AA(ProtAnalysis3D):
    """ Protocol to convert at set of carbon-alpha PDBs to all-atom PDBs using a reference all-atom PDB. """
    _label = 'c-alpha PDBs to all-atom PDBs'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('setOfPDBs', params.PointerParam, pointerClass='SetOfPDBs, SetOfAtomStructs',
                      label="Set of PDBs",
                      help='Use a scipion object SetOfPDBs / SetOfAtomStructs')
        form.addParam('aaPDB', params.PointerParam, pointerClass='AtomStruct',
                      label="All-atom pdb reference",
                      help='Use a reference all-atom PDB')

        form.addParam('useExternalCaRef', params.BooleanParam, default=False,
                      label="Uses a external C-alpha reference ?",
                      help='If yes, provides an external PDB as reference for the c-alpha model, otherwise, '
                           'uses a c-alpha-converted version of the all-atom reference')

        form.addParam('caPDB', params.PointerParam, pointerClass='AtomStruct',
                      label="c-alpha pdb reference",
                      help='Use a reference c-alpha PDB', condition="useExternalCaRef")
        form.addParam('cutoff', params.FloatParam, default=10.0,
                      label="cutoff distance (A)",
                      help='Cutoff distance used to calculate interpolation')
        form.addParam('align', params.BooleanParam, default=True,
                      label="Align references ?",
                      help='If yes, a rigid-body alignment against the reference and the PDBs to convert is performed.')

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('ca2aa')
        self._insertFunctionStep('createOutputStep')

    def ca2aa(self):

        pdbSet = self.setOfPDBs.get()
        aa_ref_pdb = self.aaPDB.get().getFileName()
        matchingType = None

        aa_ref = ContinuousFlexPDBHandler(aa_ref_pdb)
        if self.useExternalCaRef.get():
            ca_ref_pdb = self.caPDB.get().getFileName()
            ca_ref = ContinuousFlexPDBHandler(ca_ref_pdb)
        else:
            ca_ref = aa_ref.copy()
            ca_ref.select_atoms(aa_ref.allatoms2ca())
        match = aa_ref.matchPDBatoms(ca_ref, matchingType=matchingType)
        final_id = self.compute_interpolation_index(init=aa_ref, match=match, cutoff=self.cutoff.get())

        ndata = pdbSet.getSize()

        new_pdb = aa_ref.copy()
        for j in range(ndata):
            print("frame processed %i /%i " % (j + 1, ndata))
            pdbin = pdbSet[j+1].getFileName()
            pdbout = self._getExtraPath("output_%s.pdb"%str(j+1).zfill(6))
            ca_ref.coords = ContinuousFlexPDBHandler.read_coords(pdbin)

            if self.align.get():
                aa_ref = aa_ref.alignMol(ca_ref, idx_matching_atoms=match)

            vec = ca_ref.coords[match[:, 1]] - aa_ref.coords[match[:, 0]]
            for i in range(aa_ref.n_atoms):
                new_pdb.coords[i] = aa_ref.coords[i] + vec[final_id[i]].mean(axis=0)
            new_pdb.write_pdb(pdbout)

    def compute_interpolation_index(self, init, match, cutoff):
        tmp_idx = {}

        def add(dic, key, val):
            if key in dic:
                if not val in dic[key]:
                    dic[key].append(val)
                else:
                    pass
            else:
                dic[key] = [val]

        print("Computing pairlist ...")
        for i in range(init.n_atoms):
            if i % (init.n_atoms // 10) == 0:
                print("\t %i %%" % (10 * i // (init.n_atoms // 10)))
            dist_idx = match[:, 0]
            dist = np.linalg.norm(init.coords[dist_idx] - init.coords[i], axis=1)
            idx = np.where(dist < cutoff)[0]
            if len(idx) == 0:
                raise RuntimeError("At least one atoms is too far from the others with the current cutoff parameter")
            else:
                for j in idx:
                    add(tmp_idx, i, j)
                    # add(tmp_idx, j,i)
        for i in tmp_idx:
            tmp_idx[i] = np.array(tmp_idx[i])
        return tmp_idx

    def createOutputStep(self):
        pdbset = self._createSetOfPDBs("outputPDBs")
        for i in range(self.setOfPDBs.get().getSize()):
            filename = self._getExtraPath("output_%s.pdb" %str(i+1).zfill(6))
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
