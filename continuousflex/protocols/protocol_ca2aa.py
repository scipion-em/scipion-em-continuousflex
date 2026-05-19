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
    """
    Converts a collection of coarse-grained C-alpha structural models into corresponding all-atom
    structural models using a reference all-atom structure. The protocol is intended for situations
    where conformational variability has been described using simplified backbone representations,
    but subsequent biological interpretation, visualization, molecular analysis, or downstream
    structural studies require complete atomic detail.

    AI Generated:

    C-alpha to All-Atom Conversion (FlexProtCA2AA) - User Manual
        Overview

        The C-alpha to All-Atom Conversion protocol reconstructs full atomic protein models from
        C-alpha representations by using a known all-atom reference structure as a structural
        template. Its main objective is to recover atomic detail while preserving the conformational
        changes observed in a series of coarse-grained models. This approach is particularly useful
        in studies of molecular flexibility, normal mode analysis, structural interpolation, and
        continuous conformational landscapes where reduced representations are commonly employed to
        simplify calculations.

        From a biological perspective, C-alpha models are often sufficient to describe large-scale
        motions and conformational transitions, but they lack the atomic information required for
        detailed structural interpretation. Reconstructing all-atom models enables further analyses
        such as molecular visualization, interaction studies, docking experiments, residue-level
        interpretation, and preparation for molecular dynamics simulations.

        Inputs and Biological Context

        The protocol requires a collection of C-alpha structures representing different conformational
        states together with an all-atom reference structure. The reference serves as the source of
        atomic detail and defines the molecular architecture that will be transferred to the target
        conformations.

        In many practical applications, the input structures originate from flexibility analysis
        methods that generate large numbers of conformations describing molecular motion. These
        conformations often capture biologically meaningful transitions such as domain rearrangements,
        hinge motions, opening and closing events, or collective movements of macromolecular
        assemblies. The protocol enables these motions to be represented at full atomic resolution.

        Reference Selection

        Choosing an appropriate all-atom reference is one of the most important decisions for obtaining
        biologically meaningful results. The reference should correspond to the same molecular system
        represented by the C-alpha models and should contain a complete and reliable atomic
        description whenever possible.

        In some situations, an independent C-alpha reference may also be available. This can be useful
        when the coarse-grained models were generated from a specific reduced representation that does
        not perfectly match the C-alpha coordinates extracted from the all-atom structure. Using a
        dedicated C-alpha reference may improve consistency between the conformational models and the
        reconstructed atomic structures.

        Structural Alignment

        Biological structures generated from different sources or processing steps may not always share
        exactly the same coordinate system. For this reason, the protocol can perform a rigid-body
        alignment between the reference structures and the conformational models before reconstruction.

        Alignment is generally recommended when there is uncertainty regarding the relative orientation
        of the structures. Proper alignment ensures that the conformational changes are interpreted
        correctly and prevents artificial distortions in the resulting all-atom models. When all
        structures are already known to be expressed in the same coordinate frame, alignment may be
        unnecessary.

        Reconstruction Strategy

        The reconstruction process transfers atomic information from the reference structure to each
        target conformation while preserving the large-scale motions encoded in the C-alpha models.
        Nearby structural relationships within the reference are used to estimate how atomic positions
        should adapt to the new conformational state.

        A distance cutoff controls the local structural neighborhood used during this reconstruction.
        Conceptually, this parameter determines how much surrounding structural information contributes
        to the placement of atoms. Smaller values emphasize highly local structural relationships,
        whereas larger values incorporate broader structural context.

        The optimal cutoff depends on the size, flexibility, and architecture of the biological
        system. Moderate values are often suitable for most proteins, while highly flexible assemblies
        may benefit from additional testing to identify the most realistic reconstruction behavior.

        Outputs and Interpretation

        The protocol produces a new collection of all-atom structures corresponding to the input
        conformational ensemble. Each output model preserves the conformational characteristics of its
        associated C-alpha structure while providing a complete atomic description suitable for
        visualization and downstream analysis.

        These reconstructed structures can be inspected individually to study specific conformational
        states or analyzed collectively to explore molecular trajectories and structural variability.
        Because the outputs share a common atomic framework, they are particularly useful for comparing
        residue-level changes across a conformational landscape.

        Practical Recommendations

        For most biological applications, it is advisable to begin with a high-quality all-atom
        reference that closely represents the system under investigation. Visual inspection of the
        reconstructed models is recommended, especially when large conformational changes are present.

        When working with highly flexible proteins or assemblies containing multiple moving domains,
        users should verify that the reconstructed atomic models remain biologically plausible across
        the entire conformational range. Testing different cutoff values may help improve the balance
        between local structural fidelity and global conformational consistency.

        Final Perspective

        The conversion from C-alpha representations to all-atom structures bridges the gap between
        computationally efficient flexibility analyses and biologically detailed structural
        interpretation. By restoring atomic information while preserving conformational variability,
        the protocol enables researchers to move from coarse-grained descriptions of motion to
        atomically resolved models that can support visualization, mechanistic understanding, and
        further structural investigations.
    """
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
