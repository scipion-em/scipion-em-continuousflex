# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), March 2014
# * Slavica Jonic (slavica.jonic@upmc.fr)
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

from pwem.emlib import (MetaData, MDL_NMA, MDL_ENABLED, MDL_NMA_MINRANGE,
                        MDL_NMA_MAXRANGE)
from pwem.objects import AtomStruct
from pyworkflow.protocol import STEPS_PARALLEL, PointerParam, BooleanParam
from xmipp3.convert import getImageLocation
from . import FlexProtConvertToPseudoAtomsBase
from .protocol_nma_base import *
from pwem.utils import runProgram


class FlexrotNMAChoose(FlexProtConvertToPseudoAtomsBase, FlexProtNMABase):
    """
    Protocol for choosing a volume to construct an NMA analysis.

    AI Generated:

    Choose NMA (FlexrotNMAChoose) - User Manual

        Overview

        The Choose NMA protocol identifies the most representative volume from
        a collection of related three-dimensional density maps and uses it as
        the foundation for a Normal Mode Analysis (NMA) study. Its primary goal
        is to select a structural state that best captures the overall behavior
        of the dataset, providing a biologically meaningful starting point for
        exploring conformational variability and molecular flexibility.

        In structural biology projects, it is common to obtain multiple volumes
        representing different conformations, experimental conditions, or stages
        of a dynamic process. Rather than arbitrarily selecting one of these
        states for further analysis, this protocol evaluates the entire ensemble
        and determines which volume lies closest to the center of the observed
        structural landscape. The selected volume can then serve as a robust
        reference for downstream flexibility analysis.

        Inputs and General Workflow

        The protocol requires a set of volumes representing different structural
        states of the same biological system. These volumes should describe
        comparable molecular assemblies and ideally share the same sampling,
        dimensions, and overall structural content.

        Each volume is converted into a pseudoatomic representation suitable for
        Normal Mode Analysis. The resulting models provide a simplified yet
        biologically informative description of the structure that can capture
        large-scale motions while remaining computationally efficient.

        Once pseudoatomic models have been generated, a Normal Mode Analysis is
        performed independently for each candidate structure. The resulting
        normal modes describe the intrinsic directions of motion available to
        each conformation and provide the basis for comparing structural states
        across the dataset.

        Evaluating Structural Similarity

        After normal modes have been computed, each candidate structure is
        compared against every other volume in the collection. The protocol
        evaluates how well one structure can deform to resemble another using
        biologically plausible motions described by the normal modes.

        This pairwise comparison creates a global picture of structural
        relationships within the dataset. Volumes that can easily deform into
        one another are considered closely related, while larger deformations
        indicate greater structural separation.

        From a biological perspective, this analysis can reveal whether the
        dataset forms a continuous conformational spectrum or contains several
        distinct structural states. The representative volume is chosen from
        within this context rather than based on visual inspection alone.

        Optional Volume Alignment

        An optional alignment stage can be enabled before evaluating structural
        deformations. This is particularly useful when volumes may differ not
        only because of genuine conformational changes but also because of
        orientation differences introduced during reconstruction or processing.

        For datasets originating from multiple experiments, independent
        refinement procedures, or heterogeneous reconstruction pipelines,
        alignment often improves the biological interpretability of the results.
        When volumes are already expressed in a common coordinate system, this
        option may provide only limited additional benefit.

        Choosing the Representative Structure

        The central objective of the protocol is to identify the structure that
        best represents the complete ensemble. This representative model is the
        one exhibiting the smallest average deformation distance to all other
        structures in the dataset.

        Biologically, this selected volume can be interpreted as the most
        typical conformation within the observed population. It is often a
        suitable reference for flexibility studies because it minimizes bias
        toward any extreme conformational state.

        The resulting pseudoatomic model and its associated normal modes become
        the principal outputs of the protocol and can be used directly in later
        analyses involving conformational landscapes, flexible fitting, or
        motion characterization.

        Interpretation of Motion Ranges

        Beyond selecting a representative structure, the protocol estimates the
        range of observed deformations associated with each retained normal
        mode. These ranges provide an approximation of how strongly each mode
        contributes to the structural variability present in the dataset.

        Modes exhibiting broad deformation ranges may correspond to dominant
        biological motions, while modes with limited variation are generally
        less influential in explaining the observed conformational diversity.
        Such information can guide the interpretation of molecular dynamics and
        aid in selecting relevant modes for downstream exploration.

        Outputs and Their Interpretation

        The protocol produces a representative pseudoatomic model together with
        a curated set of normal modes describing its accessible motions. These
        outputs form a compact description of the structural variability
        observed across the entire collection of volumes.

        The representative model can be used as a reference structure for
        subsequent Normal Mode Analysis workflows, flexible fitting procedures,
        dimensionality reduction studies, or visualization of conformational
        transitions. The associated modes provide a biologically meaningful
        framework for understanding the dominant motions encoded in the data.

        Practical Recommendations

        The protocol performs best when all input volumes correspond to the
        same molecular assembly and differ primarily because of conformational
        variability. Large differences arising from reconstruction artifacts,
        inconsistent preprocessing, or unrelated biological states can reduce
        the reliability of the representative selection.

        Careful preparation of the input dataset is therefore important.
        Volumes should be inspected to ensure consistency in scale, sampling,
        and molecular content before analysis. When substantial orientation
        differences are expected, enabling alignment is generally advisable.

        Final Perspective

        Choosing an appropriate reference structure is a critical step in many
        flexibility analysis workflows. By identifying the volume that best
        represents the overall conformational ensemble, this protocol provides
        a principled and biologically meaningful foundation for Normal Mode
        Analysis. The resulting model and motion descriptors help transform a
        collection of individual structural states into a coherent description
        of molecular dynamics and functional flexibility.
    """
    _label = 'choose NMA'

    def __init__(self, **args):
        FlexProtConvertToPseudoAtomsBase.__init__(self, **args)
        FlexrotNMABase.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputStructures', PointerParam, label="Input volumes",
                      important=True,
                      pointerClass='SetOfVolumes')
        form.addParam('alignVolumes', BooleanParam, label="Align volumes",
                      default=False,
                      help="Align deformed PDBs to volume to maximize match")
        FlexProtConvertToPseudoAtomsBase._defineParams(self, form)
        form.addParallelSection(threads=4, mpi=1)

        form.addSection(label='Normal Mode Analysis')
        FlexProtNMABase._defineParamsCommon(self, form)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        inputStructures = self.inputStructures.get()
        self.sampling = inputStructures.getSamplingRate()
        filenames = []
        for inputStructure in inputStructures:
            filenames.append(getImageLocation(inputStructure))

        deps = []
        for volCounter in range(1, len(filenames) + 1):
            fnIn = filenames[volCounter - 1]
            prefix = "_%02d" % volCounter
            fnMask = self._insertMaskStep(fnIn, prefix)

            self._insertFunctionStep('convertToPseudoAtomsStep', inputStructure,
                                     fnIn, fnMask, prefix, prerequisites=deps)
            parentId = self._insertFunctionStep('computeNMAStep', self._getPath(
                "pseudoatoms%s.pdb" % prefix), prefix)
            deps = []
            for volCounter2 in range(1, len(filenames) + 1):
                if volCounter2 != volCounter:
                    args = "-i %s --pdb %s --modes %s --sampling_rate %f -o %s --fixed_Gaussian %f --opdb %s" % \
                           (filenames[volCounter2 - 1],
                            self._getPath("pseudoatoms%s.pdb" % prefix), \
                            self._getPath("modes%s.xmd" % prefix),
                            self.sampling, \
                            self._getExtraPath('alignment_%02d_%02d.xmd' % (
                            volCounter, volCounter2)), \
                            self.sampling * self.pseudoAtomRadius.get(),
                            self._getExtraPath('alignment_%02d_%02d.pdb' % (
                            volCounter, volCounter2)))
                    if self.alignVolumes.get():
                        args += " --alignVolumes"
                    stepId = self._insertRunJobStep("xmipp_nma_alignment_vol",
                                                    args,
                                                    prerequisites=[parentId])
                    deps.append(stepId)

        self._insertFunctionStep('evaluateDeformationsStep', prerequisites=deps)

    # --------------------------- Step functions --------------------------------------------
    def convertToPseudoAtomsStep(self, inputStructure, fnIn, fnMask, prefix):
        FlexProtConvertToPseudoAtomsBase.convertToPseudoAtomsStep(self, fnIn,
                                                                   fnMask,
                                                                   prefix)
        self.createChimeraScriptStep(inputStructure, fnIn, prefix)
        createLink(self._getPath("pseudoatoms%s.pdb" % prefix),
                   self._getPath("pseudoatoms.pdb"))

    def computeNMAStep(self, fnIn, prefix):
        cutoffStr = ''
        if self.cutoffMode == NMA_CUTOFF_REL:
            cutoffStr = 'Relative %f' % self.rcPercentage.get()
        else:
            cutoffStr = 'Absolute %f' % self.rc.get()
        self.computeModesStep(fnIn, self.numberOfModes.get(), cutoffStr)
        self.reformatOutputStep("pseudoatoms.pdb")
        self.qualifyModesStep(self.numberOfModes.get(),
                              self.collectivityThreshold.get(), True)
        fnModes = self._getPath("modes.xmd")
        fnModesPrefix = self._getPath("modes%s.xmd" % prefix)
        runProgram("xmipp_metadata_utilities",
                    "-i %s --operate modify_values \"nmaModeFile=replace(nmaModeFile,'/modes/','/modes%s/')\" -o %s" %
                    (fnModes, prefix, fnModesPrefix))
        runProgram("mv", "%s %s" % (
        self._getPath('modes'), self._getPath('modes%s' % prefix)))

        # Remove intermediate files
        cleanPath(self._getPath("pseudoatoms.pdb"), fnModes,
                  self._getExtraPath('vec_ani.pkl'))

    def evaluateDeformationsStep(self):
        N = self.inputStructures.get().getSize()
        import numpy
        distances = numpy.zeros([N, N])
        for volCounter in range(1, N + 1):
            pdb1 = open(
                self._getPath('pseudoatoms_%02d.pdb' % volCounter)).readlines()
            for volCounter2 in range(1, N + 1):
                if volCounter != volCounter2:
                    davg = 0.
                    Navg = 0.
                    pdb2 = open(self._getExtraPath('alignment_%02d_%02d.pdb' % (
                    volCounter, volCounter2))).readlines()
                    for i in range(len(pdb1)):
                        line1 = pdb1[i]
                        if line1.startswith("ATOM"):
                            line2 = pdb2[i]
                            x1 = float(line1[30:37])
                            y1 = float(line1[38:45])
                            z1 = float(line1[46:53])
                            x2 = float(line2[30:37])
                            y2 = float(line2[38:45])
                            z2 = float(line2[46:53])
                            dx = x1 - x2
                            dy = y1 - y2
                            dz = z1 - z2
                            d = math.sqrt(dx * dx + dy * dy + dz * dz)
                            davg += d
                            Navg += 1
                    if Navg > 0:
                        davg /= Navg
                    distances[volCounter - 1, volCounter2 - 1] = davg
        distances = 0.5 * (distances + numpy.transpose(distances))
        numpy.savetxt(self._getPath('distances.txt'), distances)
        distances1D = numpy.mean(distances, axis=0)
        print("Average distance to rest of volumes=", distances1D)
        imin = numpy.argmin(distances1D)
        print("The volume in the middle is pseudoatoms_%02d.pdb" % (imin + 1))
        createLink(self._getPath("pseudoatoms_%02d.pdb" % (imin + 1)),
                   self._getPath("pseudoatoms.pdb"))
        createLink(self._getPath("modes_%02d.xmd" % (imin + 1)),
                   self._getPath("modes.xmd"))
        createLink(
            self._getExtraPath("pseudoatoms_%02d_distance.hist" % (imin + 1)),
            self._getExtraPath("pseudoatoms_distance.hist"))

        # Measure range
        minDisplacement = 1e38 * numpy.ones([self.numberOfModes.get(), 1])
        maxDisplacement = -1e38 * numpy.ones([self.numberOfModes.get(), 1])
        mdNMA = MetaData(self._getPath("modes.xmd"))
        for volCounter in range(1, N + 1):
            if volCounter != imin + 1:
                md = MetaData(self._getExtraPath(
                    "alignment_%02d_%02d.xmd" % (imin + 1, volCounter)))
                displacements = md.getValue(MDL_NMA, md.firstObject())
                idx1 = 0
                idx2 = 0
                for idRow in mdNMA:
                    if mdNMA.getValue(MDL_ENABLED, idRow) == 1:
                        minDisplacement[idx2] = min(minDisplacement[idx2],
                                                    displacements[idx1])
                        maxDisplacement[idx2] = max(maxDisplacement[idx2],
                                                    displacements[idx1])
                        idx1 += 1
                    else:
                        minDisplacement[idx2] = 0
                        maxDisplacement[idx2] = 0
                    idx2 += 1
        idx2 = 0
        for idRow in mdNMA:
            mdNMA.setValue(MDL_NMA_MINRANGE, float(minDisplacement[idx2]),
                           idRow)
            mdNMA.setValue(MDL_NMA_MAXRANGE, float(maxDisplacement[idx2]),
                           idRow)
            idx2 += 1
        mdNMA.write(self._getPath("modes.xmd"))

        # Create output
        volCounter = 0
        for inputStructure in self.inputStructures.get():
            if volCounter == imin:
                print("The corresponding volume is %s" % (
                    getImageLocation(inputStructure)))
                finalStructure = inputStructure
                break
            volCounter += 1

        pdb = AtomStruct(self._getPath('pseudoatoms.pdb'), pseudoatoms=True)
        self._defineOutputs(outputPdb=pdb)
        modes = NormalModes(filename=self._getPath('modes.xmd'))
        self._defineOutputs(outputModes=modes)

        self._defineSourceRelation(self.inputStructures, self.outputPdb)
        # ToDo: the self.outputPdb should be a Pointer, not an object

    #         self._defineSourceRelation(self.outputPdb, self.outputModes)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append(
            'Pseudoatom radius (voxels): %f' % self.pseudoAtomRadius.get())
        summary.append(
            'Approximation target error (%%): %f' % self.pseudoAtomTarget.get())
        return summary

    def _methods(self):
        summary = []
        #        summary.append('We converted the volume %s into a pseudoatomic representation with Gaussian atoms (sigma=%f A and a target error'\
        #                       ' of %f%%) [Nogales2013].'%(self.inputStructure.get().getNameId(),
        #                                     self.pseudoAtomRadius.get()*self.inputStructure.get().getSamplingRate(),
        #                                     self.pseudoAtomTarget.get()));
        #        if self.hasAttribute('outputPdb'):
        #            summary.append('We refer to the pseudoatomic model as %s.'%self.outputPdb.getNameId())
        return summary

    def _citations(self):
        return ['harastani2022continuousflex','Nogales2013', 'Jin2014']
