# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
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

import os
import time

import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from pyworkflow.protocol.params import NumericRangeParam
from pyworkflow.utils import getListFromRangeString
from pyworkflow.utils import replaceExt
import numpy as np
from math import cos, sin, pi
from pwem.objects import AtomStruct

from continuousflex.protocols.utilities.pdb_handler import ContinuousFlexPDBHandler

MODE_RELATION_LINEAR = 0
MODE_RELATION_3CLUSTERS = 1
MODE_RELATION_MESH = 2
MODE_RELATION_RANDOM = 3
MODE_RELATION_PARABOLA = 4


class FlexProtSynthesizePDBs(ProtAnalysis3D):
    """
    Synthesizes ensembles of flexible atomic structures from a reference
    model and a set of normal modes. The protocol generates multiple PDB
    conformations that represent structural variability along selected
    collective motions and can be used as ground-truth data for method
    development, validation, simulation, or heterogeneity studies.

    AI Generated:

    Synthesize PDBs (FlexProtSynthesizePDBs) - User Manual
        Overview

        The Synthesize PDBs protocol generates collections of atomic
        structures that represent different conformational states of a
        macromolecule. It uses previously computed normal modes to deform
        a reference structure and produce realistic structural variations
        that follow biologically meaningful collective motions. The
        resulting ensemble can serve as a synthetic representation of
        conformational landscapes observed in flexible proteins, molecular
        machines, or large biological assemblies.

        For structural biology applications, this protocol is particularly
        useful when studying flexibility, benchmarking computational
        methods, generating controlled datasets, or exploring how
        structural variability may appear in downstream analyses. The
        generated structures preserve the overall architecture of the
        original model while introducing controlled conformational changes
        guided by normal mode analysis.

        Inputs and General Workflow

        The protocol requires a set of normal modes associated with a
        molecular structure. These modes describe collective directions of
        motion that are often related to biologically relevant
        conformational transitions. The user selects which modes will be
        used to generate variability and defines how amplitudes along
        those modes should be distributed.

        The selected modes determine the deformation space explored by the
        generated structures. In many practical applications, a small
        number of low-frequency modes is sufficient to capture large-scale
        conformational changes. These motions often correspond to domain
        movements, opening and closing transitions, or rearrangements that
        are difficult to observe using static structures alone.

        Defining Relationships Between Modes

        A key aspect of the protocol is the ability to control the
        relationship between mode amplitudes. Different relationships
        generate different types of conformational landscapes and can be
        chosen according to the biological or methodological objective.

        The Linear relationship generates structures along a coordinated
        trajectory where selected modes vary together. This approach is
        useful when modeling a single dominant transition between
        conformational states or when creating datasets with a simple and
        interpretable variability pattern.

        The Three Clusters option creates three well-separated groups of
        conformations. This configuration is particularly useful for
        testing classification, clustering, and heterogeneity analysis
        methods because the resulting structures naturally form discrete
        conformational classes.

        The Grid option systematically samples combinations of amplitudes
        across two selected modes. The resulting ensemble covers the
        deformation space in an organized manner and is useful for
        visualizing energy landscapes, evaluating dimensionality reduction
        methods, or studying continuous transitions between states.

        The Random option generates conformations with independently
        sampled amplitudes. This approach produces broad structural
        diversity and is often suitable for benchmarking algorithms under
        heterogeneous conditions.

        The Parabolic option constrains the generated conformations to
        follow a curved trajectory in deformation space. Such datasets are
        valuable when investigating nonlinear conformational variability
        and assessing methods designed to capture continuous but
        non-Euclidean structural relationships.

        Amplitude Selection and Conformational Diversity

        The amplitude range controls the extent of structural deformation.
        Small amplitudes generally produce conformations close to the
        reference structure, whereas larger amplitudes generate more
        pronounced motions and greater diversity.

        From a biological perspective, excessively large amplitudes may
        produce structures that move beyond physically realistic
        conformations. Users should therefore choose amplitude ranges that
        remain consistent with the expected flexibility of the system under
        study. Moderate values are often appropriate when the objective is
        to model plausible conformational transitions.

        Number of Generated Structures

        The protocol allows the generation of a user-defined number of
        conformations. Larger ensembles provide denser sampling of the
        conformational landscape and are often beneficial when producing
        benchmark datasets or studying continuous heterogeneity.

        When the Grid relationship is selected, the number of generated
        structures is determined by the sampling density of the grid. A
        finer grid provides more detailed coverage of the deformation
        space but increases storage and computational requirements.

        Reproducibility and Randomization

        The protocol supports both randomized and reproducible generation
        strategies. Randomized generation creates different ensembles in
        different executions, which is useful for exploring variability
        and producing independent datasets.

        Reproducible generation ensures that identical conformations are
        created across repeated runs. This capability is particularly
        important for benchmarking studies, algorithm comparisons, and
        controlled experiments where consistency between datasets must be
        maintained.

        Outputs and Their Interpretation

        The primary output is a set of PDB structures representing
        different conformational states of the same molecular system. Each
        generated structure corresponds to a specific position within the
        selected deformation space and can be used independently or as
        part of the complete ensemble.

        The generated collection can be employed in downstream workflows
        such as volume synthesis, simulation studies, heterogeneity
        analysis, machine learning dataset creation, or validation of
        reconstruction and classification methods. Because the
        conformational coordinates are known, the resulting datasets are
        particularly valuable as reference standards for methodological
        development.

        Practical Recommendations

        For most biological applications, selecting a small number of
        dominant low-frequency modes provides the most interpretable and
        realistic conformational variability. Linear or Grid sampling is
        often preferred when studying continuous transitions, whereas
        Three Clusters is advantageous when evaluating classification
        performance.

        Random sampling is useful when broad structural diversity is
        desired, while Parabolic sampling can reveal the behavior of
        methods under nonlinear variability. Regardless of the chosen
        strategy, users should verify that the generated conformations
        remain biologically plausible and consistent with known structural
        constraints.

        Final Perspective

        For researchers interested in molecular flexibility, this
        protocol provides a controlled framework for generating synthetic
        conformational ensembles directly from normal mode analysis. By
        transforming a static structure into a collection of related
        states, it enables systematic investigation of structural
        heterogeneity and supports the development and validation of
        advanced computational methods for cryo-EM and structural biology.
    """
    _label = 'synthesize PDBs'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection('Input')
        form.addParam('inputModes', params.PointerParam,
                      pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', NumericRangeParam,
                      label="Modes selection",
                      allowsNull=True,
                      default='7-8',
                      help='Select the normal modes that will be used for PDB synthesis.'
                           'It is usually two modes that should be selected, unless if the relationship is linear or '
                           'random.\n '
                           'You have several ways to specify the modes.\n'
                           'Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('modeRelationChoice', params.EnumParam, default=MODE_RELATION_LINEAR,
                      choices=['Linear relationship', '3 clusters', 'Grid', 'Random', 'Parabolic'],
                      label='Relationship between the modes',
                      help='linear relationship: all the selected modes will have equal amplitudes. \n'
                           '3 clusters: the volumes will be divided exactly into three classes.\n'
                           'Grid: the amplitudes will be in a grid shape (grid size is square of what grid step).\n'
                           'Random: all the amplitudes will be random in the given range.\n'
                           'Parabolic: The relationship between two modes will resembles an upper half circle.')
        form.addParam('centerPoint', params.IntParam, default=100,
                      condition='modeRelationChoice==%d' % MODE_RELATION_3CLUSTERS,
                      label='Center point',
                      help='This number will be used to determine the distance between the clusters'
                           'center1 = (-center_point, 0)'
                           'center2 = (center_point, 0)'
                           'center3 = (0, center_point)')
        form.addParam('modesAmplitudeRange', params.IntParam, default=150,
                      allowsNull=True,
                      condition='modeRelationChoice != %d' % MODE_RELATION_3CLUSTERS,
                      label='Amplitude range N --> [-N, N]',
                      help='Choose the number N for which the generated normal mode amplitudes are in the range of'
                           ' [-N, N]')
        form.addParam('meshRowPoints', params.IntParam, default=6,
                      allowsNull=True,
                      condition='modeRelationChoice==%d' % MODE_RELATION_MESH,
                      label='Grid number of steps',
                      help='This number will be the number of points in the row and the column (grid shape will be '
                           'size*size)')
        form.addParam('numberOfPDBs', params.IntParam, default=36,
                      label='Number of PDBs',
                      condition='modeRelationChoice!=%d' % MODE_RELATION_MESH,
                      help='Number of atomic structures that will be generated')
        form.addParam('seedOption', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Random seed',
                      help='Keeping it as True means that different runs will generate different PDBs in terms '
                           'of conformations. If you set as False, then different runs will '
                           'have the same conformations and angles '
                           '(setting to False allows you to generate the same conformations and orientations with '
                           'different noise values).')

        # form.addParallelSection(threads=0, mpi=8)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        if self.seedOption.get():
            np.random.seed(int(time.time()))
        else:
            np.random.seed(0)
        self._insertFunctionStep("generateDeformationsStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions --------------------------------------------
    def generateDeformationsStep(self):
        # Find the right PDB file to use for data synthesis
        pdb_name1 = os.path.dirname(self.inputModes.get().getFileName()) + '/atoms.pdb'
        pdb_name2 = os.path.dirname(self.inputModes.get().getFileName()) + '/pseudoatoms.pdb'
        if os.path.exists(pdb_name1):
            fnPDB = pdb_name1
        else:
            fnPDB = pdb_name2
        # use the input relationship between the modes to generate normal mode amplitudes metadata
        fnModeList = replaceExt(self.inputModes.get().getFileName(), 'xmd')
        modeAmplitude = self.modesAmplitudeRange.get()
        meshRowPoints = self.meshRowPoints.get()
        numberOfModes = self.inputModes.get().getSize()
        modeSelection = np.array(getListFromRangeString(self.modeList.get()))
        deformationFile = self._getExtraPath('GroundTruth.xmd')
        pdbMD = md.MetaData()
        # these vairables for alternating the generation when 3 clusters are selected
        cluster1 = cluster2 = cluster3 = False
        # these variables for the generaton of a mesh if selected
        XX, YY = np.meshgrid(np.linspace(start=-modeAmplitude, stop=modeAmplitude, num=meshRowPoints),
                             np.linspace(start=modeAmplitude, stop=-modeAmplitude, num=meshRowPoints))
        mode7_samples = XX.reshape(-1)
        mode8_samples = YY.reshape(-1)

        # iterate over the number of outputs (if mesh, this has to be calculated)
        numberOfPDBs = self.getNumberOfPdbs()
        def readModes(fnIn):
            modesMD = md.MetaData(fnIn)
            vectors = []
            for objId in modesMD:
                vecFn = modesMD.getValue(md.MDL_NMA_MODEFILE, objId)
                vec = np.loadtxt(vecFn)
                vectors.append(vec)
            return vectors

        pdb = ContinuousFlexPDBHandler(fnPDB)
        modes = readModes(fnModeList)

        for i in range(numberOfPDBs):
            deformations = np.zeros(numberOfModes)

            if self.modeRelationChoice == MODE_RELATION_LINEAR:
                amplitude = self.modesAmplitudeRange.get()
                deformations[modeSelection - 1] = np.ones(len(modeSelection)) * np.random.uniform(-amplitude, amplitude)
            elif self.modeRelationChoice == MODE_RELATION_3CLUSTERS:
                center_point = self.centerPoint.get()
                center1 = (-center_point, 0)
                center2 = (center_point, 0)
                center3 = (0, center_point)
                if not (cluster1 or cluster2 or cluster3):
                    cluster1 = True
                if cluster3:
                    deformations[modeSelection - 1] = center3
                    cluster3 = False
                if cluster2:
                    deformations[modeSelection - 1] = center2
                    cluster2 = False
                    cluster3 = True
                if cluster1:
                    deformations[modeSelection - 1] = center1
                    cluster1 = False
                    cluster2 = True
            elif self.modeRelationChoice == MODE_RELATION_RANDOM:
                amplitude = self.modesAmplitudeRange.get()
                deformations[modeSelection - 1] = np.random.uniform(-amplitude, amplitude, len(modeSelection))
            elif self.modeRelationChoice == MODE_RELATION_PARABOLA:
                amplitude = self.modesAmplitudeRange.get()
                rv = np.random.uniform(0, 1)
                point = (amplitude * cos(rv * pi), amplitude * sin(rv * pi))
                deformations[modeSelection - 1] = point

            elif self.modeRelationChoice == MODE_RELATION_MESH:
                new_point = (mode7_samples[i], mode8_samples[i])
                deformations[modeSelection - 1] = new_point

            # we won't keep the first 6 modes
            deformations = deformations[6:]

            self.nma_deform_pdb(pdb.copy(), modes, self._getExtraPath(str(i + 1).zfill(5) + '_df.pdb'), deformations)

            pdbMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i + 1).zfill(5) + '_df.pdb'),
                           pdbMD.addObject())
            pdbMD.setValue(md.MDL_NMA, list(deformations), i + 1)

        pdbMD.write(deformationFile)

    def nma_deform_pdb(self, pdb, modes, fnOut, deformList):
        for i in range(len(deformList)):
            pdb.coords += deformList[i] * modes[7 - 1 + i]
        pdb.write_pdb(fnOut)

    def createOutputStep(self):
        pdbset = self._createSetOfPDBs("outputPDBs")
        pdbMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        for objID in pdbMD:
            filename = pdbMD.getValue(md.MDL_IMAGE, objID)
            pdb = AtomStruct(filename=filename)
            pdbset.append(pdb)

        self._defineOutputs(outputPDBs=pdbset)

    def getNumberOfPdbs(self):
        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfPDBs = self.meshRowPoints.get() ** 2
        else:
            numberOfPDBs = self.numberOfPDBs.get()
        return numberOfPDBs

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """
        return self.inputModes.get().getPdb()

    # --------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd', 
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    def _getLocalModesFn(self):
        modesFn = self.inputModes.get().getFileName()
        return self._getBasePath(modesFn)

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
