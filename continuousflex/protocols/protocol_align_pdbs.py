# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Remi Vuillemot             (remi.vuillemot@upmc.fr)
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
from xmipp3.convert import writeSetOfVolumes, writeSetOfParticles, readSetOfVolumes, readSetOfParticles
from pwem.constants import ALIGN_PROJ
from continuousflex.protocols.convert import matrix2eulerAngles

import numpy as np
import glob
import pwem.emlib.metadata as md

PDB_SOURCE_PATTERN = 0
PDB_SOURCE_OBJECT = 1
PDB_SOURCE_TRAJECT = 2

MATCHING_PDB_NONE = 0
MATCHING_PDB_CHAIN = 1
MATCHING_PDB_SEG = 2

class FlexProtAlignPdb(ProtAnalysis3D):
    """
    Performs rigid-body alignment of atomic structures represented as
    PDB files or molecular dynamics trajectories. The protocol places
    multiple structural conformations into a common coordinate system,
    enabling direct structural comparison, visualization, variability
    analysis, and integration with downstream cryo-EM or structural
    biology workflows.

    AI Generated:

    PDB Rigid Body Alignment (FlexProtAlignPdb) - User Manual
        Overview

        The PDB Rigid Body Alignment protocol aligns a collection of
        atomic structures to a selected reference structure using
        rigid-body transformations. Its primary objective is to remove
        differences caused by overall rotation and translation so that
        biologically meaningful conformational variations can be studied
        in a consistent spatial frame.

        In structural biology projects, ensembles of structures may
        originate from molecular dynamics simulations, normal mode
        analyses, flexible fitting procedures, integrative modeling, or
        collections of experimentally determined conformations. Before
        these structures can be compared quantitatively, they must be
        expressed within a common coordinate system. This protocol
        provides that standardization step and facilitates subsequent
        analyses focused on molecular flexibility and structural
        heterogeneity.

        Inputs and General Workflow

        The protocol accepts structural data from several sources. Users
        may provide a collection of PDB files, an existing set of
        atomic structures stored within a project, or molecular dynamics
        trajectory files accompanied by an appropriate structural
        reference. This flexibility allows the protocol to be used both
        for static structural ensembles and for large conformational
        trajectories.

        A reference structure is required to define the target
        coordinate system. All input conformations are aligned against
        this reference so that equivalent structural regions occupy the
        same spatial frame. Choosing a biologically representative and
        well-curated reference generally improves the interpretability
        of the resulting aligned ensemble.

        Structural Correspondence Between Models

        One of the most important considerations when aligning atomic
        structures is determining which atoms should be considered
        equivalent between the reference and the input structures. In
        simple cases, all structures may already share identical atom
        ordering and composition, allowing direct alignment.

        More complex datasets may contain differences in atom ordering,
        chain organization, or segmentation conventions. The protocol
        supports correspondence strategies based on chain identity or
        segment identity together with residue numbering. These options
        help ensure that equivalent biological regions are compared even
        when file organization differs between structures.

        From a biological perspective, careful correspondence selection
        is critical. Incorrect matching can produce alignments that are
        geometrically valid but biologically meaningless.

        Working with Molecular Dynamics Trajectories

        The protocol is particularly useful for molecular dynamics
        studies where thousands of conformations may be generated during
        a simulation. In these situations, users can analyze only a
        selected portion of the trajectory by specifying a starting
        frame, ending frame, and sampling interval.

        This capability allows researchers to focus on equilibrated
        regions of a simulation, reduce computational cost, or study
        specific conformational transitions. Sampling trajectories at
        regular intervals is often sufficient to capture large-scale
        motions while avoiding unnecessary redundancy.

        Alignment and Biological Interpretation

        Rigid-body alignment removes global motion while preserving
        internal structural differences. As a result, conformational
        changes observed after alignment are more likely to reflect
        biologically relevant flexibility rather than arbitrary
        differences in molecular orientation.

        This distinction is particularly important when studying domain
        movements, allosteric transitions, hinge motions, or ensemble
        variability. By eliminating overall translation and rotation,
        researchers can focus on the structural changes that are most
        relevant to biological function.

        Outputs and Their Interpretation

        The protocol produces an aligned structural ensemble expressed
        in the coordinate system of the selected reference. The aligned
        structures can be inspected visually, used for statistical
        analyses, or incorporated into additional flexibility studies.

        Alignment parameters describing the rigid-body transformations
        are also generated. These transformations provide a compact
        representation of the spatial relationship between each
        conformation and the reference structure.

        Optional Application to Other Data

        In many cryo-EM and integrative structural biology workflows,
        structural models are associated with other experimental data
        such as particle images or reconstructed volumes. The protocol
        can propagate the computed rigid-body transformations to these
        associated datasets, ensuring that all related information is
        represented within a consistent coordinate system.

        This capability is particularly useful when combining atomic
        models with volumetric maps, subtomograms, or particle datasets,
        allowing structural and imaging information to remain
        synchronized throughout subsequent analyses.

        Practical Recommendations

        For most applications, users should select a reference
        structure that represents the dominant or most biologically
        relevant conformation. When structures originate from different
        sources or processing pipelines, verifying atom correspondence
        before alignment is strongly recommended.

        For molecular dynamics trajectories, it is often beneficial to
        exclude non-equilibrated regions and analyze representative
        frames. When large conformational changes are expected, visual
        inspection of the aligned ensemble can help distinguish genuine
        biological motions from artifacts arising from incomplete atom
        correspondence.

        Final Perspective

        Rigid-body alignment is a foundational step in the analysis of
        structural ensembles. By placing all conformations into a common
        spatial frame, the protocol enables meaningful comparison of
        molecular states, supports quantitative studies of flexibility,
        and facilitates integration between atomic models and cryo-EM
        data. Careful selection of the reference structure and proper
        definition of atomic correspondence are the key factors for
        obtaining biologically reliable results.
    """
    _label = 'pdbs rigid body alignment'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('pdbSource', EnumParam, default=PDB_SOURCE_PATTERN,
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
                      label="Alignment Reference PDB",
                      help='Reference PDB to align the PDBs with')
        form.addParam('matchingType', params.EnumParam, label="Match PDBs and reference PDB ?", default=MATCHING_PDB_NONE,
                      choices=['All PDBs are matching', 'Match chain name + residue no',
                               'Match segment name + residue no'],
                      help="Method to find atomic coordinates correspondence between the pdb set "
                           "coordinates and the reference PDB. The method will select the matching atoms"
                           " and sort them in the corresponding order. If the structures in the files are"
                           " already matching, choose All structures are matching")

        form.addParam('createOutput', params.BooleanParam, default=True,
                      label="Create output Set of PDBs ?",
                      help='Create output set. This step can be time consuming and not necessary if you are only '
                           ' interested by the alignment parameters. The aligned coordinate are conserved as DCD file '
                           'in the extra directory.'
                        , expertLevel=params.LEVEL_ADVANCED)

        form.addSection(label='Apply alignment to other set')
        form.addParam('applyAlignment', params.BooleanParam, default=False,
                      label="Apply alignment to other data set ?",
                      help='Use the PDB alignment to align another data set.')
        form.addParam('otherSet', params.PointerParam, pointerClass='SetOfParticles, SetOfVolumes',
                      condition='applyAlignment',
                      label="Other set of Particles / Volumes",
                      help='Use a scipion EMSet object')



        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readInputFiles')
        self._insertFunctionStep('rigidBodyAlignmentStep')
        if self.applyAlignment.get():
            self._insertFunctionStep('applyAlignmentStep')
        if self.createOutput.get():
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

    def rigidBodyAlignmentStep(self):

        # open files
        inputPDB = ContinuousFlexPDBHandler(self.getPDBRef())
        inputPDB.write_pdb(self._getExtraPath("reference.pdb"))
        refPDB = ContinuousFlexPDBHandler(self.alignRefPDB.get().getFileName())
        arrDCD = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ =arrDCD.shape
        alignXMD = md.MetaData()

        # find matching index between reference and pdbs
        if self.matchingType.get() == MATCHING_PDB_CHAIN:
            idx_matching_atoms = inputPDB.matchPDBatoms(reference_pdb=refPDB, matchingType=0)
            refPDB.select_atoms(idx_matching_atoms[:, 1])
        elif self.matchingType.get() == MATCHING_PDB_SEG:
            idx_matching_atoms = inputPDB.matchPDBatoms(reference_pdb=refPDB, matchingType=1)
            refPDB.select_atoms(idx_matching_atoms[:, 1])
        else:
            idx_matching_atoms = None

        # loop over all pdbs
        for i in range(nframe):
            print("Aligning PDB %i ... " %i)

            # rotate
            if self.matchingType.get() != MATCHING_PDB_NONE :
                coord = arrDCD[i][idx_matching_atoms[:, 0]]
            else:
                coord = arrDCD[i]
            rot_mat, tran = ContinuousFlexPDBHandler.alignCoords(refPDB.coords, coord)
            arrDCD[i] = (np.dot(arrDCD[i], rot_mat) + tran).astype(np.float32)

            # add to MD
            trans_mat = np.zeros((4,4))
            trans_mat[:3,:3] = rot_mat
            trans_mat[:,3][:3] = tran
            rot, tilt, psi,shftx, shfty, shftz = matrix2eulerAngles(trans_mat)
            index = alignXMD.addObject()
            alignXMD.setValue(md.MDL_ANGLE_ROT, rot, index)
            alignXMD.setValue(md.MDL_ANGLE_TILT, tilt, index)
            alignXMD.setValue(md.MDL_ANGLE_PSI, psi, index)
            alignXMD.setValue(md.MDL_SHIFT_X, shftx, index)
            alignXMD.setValue(md.MDL_SHIFT_Y, shfty, index)
            alignXMD.setValue(md.MDL_SHIFT_Z, shftz, index)
            alignXMD.setValue(md.MDL_IMAGE, "", index)

        numpyArr2dcd(arrDCD, self._getExtraPath("coords.dcd"))
        alignXMD.write(self._getExtraPath("alignment.xmd"))


    def createOutputStep(self):
        pdbset = self._createSetOfPDBs("outputPDBs")
        arrDCD = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        refPDB = ContinuousFlexPDBHandler(self.getPDBRef())

        nframe, natom,_ = arrDCD.shape
        for i in range(nframe):
            filename = self._getExtraPath("output_%s.pdb" %str(i+1).zfill(6))
            refPDB.coords = arrDCD[i]
            refPDB.write_pdb(filename)
            pdb = AtomStruct(filename=filename)
            pdbset.append(pdb)

        self._defineOutputs(outputPDBs=pdbset)

    def applyAlignmentStep(self):
        inputSet = self.otherSet.get()

        if isinstance(inputSet, SetOfVolumes):
            inputAlignment = self._createSetOfVolumes("inputAlignment")
            readSetOfVolumes(self._getExtraPath("alignment.xmd"), inputAlignment)
            alignedSet = self._createSetOfVolumes("alignedSet")
        else:
            inputAlignment = self._createSetOfParticles("inputAlignment")
            alignedSet = self._createSetOfParticles("alignedSet")
            readSetOfParticles(self._getExtraPath("alignment.xmd"), inputAlignment)

        alignedSet.setSamplingRate(inputSet.getSamplingRate())
        alignedSet.setAlignment(ALIGN_PROJ)
        iter1 = inputSet.iterItems()
        iter2 = inputAlignment.iterItems()
        for i in range(inputSet.getSize()):
            p1 = iter1.__next__()
            p2 = iter2.__next__()
            r1 = p1.getTransform()
            r2 = p2.getTransform()
            rot = r2.getRotationMatrix()
            tran = np.array(r2.getShifts()) / inputSet.getSamplingRate()
            new_trans = np.zeros((4, 4))
            new_trans[:3, 3] = tran
            new_trans[:3, :3] = rot
            new_trans[3, 3] = 1.0
            r1.composeTransform(new_trans)
            p1.setTransform(r1)
            alignedSet.append(p1)
        self._defineOutputs(alignedSet = alignedSet)

        if isinstance(inputSet, SetOfVolumes):
            writeSetOfVolumes(alignedSet, self._getExtraPath("alignedSet.xmd"))
        else:
            writeSetOfParticles(alignedSet, self._getExtraPath("alignedSet.xmd"))
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
