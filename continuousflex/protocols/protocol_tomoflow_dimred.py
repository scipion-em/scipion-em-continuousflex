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
    """
    Reduces the dimensionality of deformation information derived from
    optical flow analysis of 3D volumes, enabling the exploration and
    visualization of structural variability in a compact and interpretable
    space.

    AI Generated:

    Heterogeneous Flow Dimensionality Reduction (FlexProtDimredHeteroFlow) - User Manual

        Overview

        The Heterogeneous Flow Dimensionality Reduction protocol is designed
        to simplify the analysis of complex conformational variability
        captured through optical flow measurements between three-dimensional
        volumes. In structural biology studies, optical flow data often
        describe high-dimensional deformation patterns that are difficult to
        interpret directly. This protocol transforms those deformation
        descriptors into a lower-dimensional representation while preserving
        the most meaningful relationships between samples.

        For biological users, the main objective is to reveal the underlying
        organization of conformational landscapes. By projecting deformation
        information into two or a few dimensions, the protocol allows the
        identification of structural continua, conformational clusters, rare
        states, and transition pathways that may otherwise remain hidden in
        the original high-dimensional space.

        Inputs and General Workflow

        The protocol requires as input a previous optical flow analysis in
        which a collection of volumes has been compared against a reference
        structure. These deformation measurements represent the structural
        differences between individual volumes and the chosen reference
        state.

        The dimensionality reduction process converts these deformation
        descriptors into a compact coordinate system. Each volume is then
        represented by a small number of variables that summarize its
        position within the overall conformational landscape. The resulting
        coordinates can be used for visualization, clustering, classification,
        trajectory analysis, or as input for additional computational methods.

        Understanding Dimensionality Reduction

        Biological systems often exhibit complex motions involving many
        degrees of freedom. Although deformation measurements may contain a
        large number of variables, the biologically relevant motions are
        frequently governed by a much smaller set of collective movements.
        Dimensionality reduction aims to identify these dominant patterns.

        In practice, the reduced representation can help distinguish
        different functional states, identify intermediate conformations, or
        reveal continuous motions connecting multiple structural forms. The
        reduced coordinates should not be interpreted as direct physical
        quantities but rather as abstract descriptors capturing major sources
        of variability within the dataset.

        Choice of Dimensionality Reduction Method

        The protocol offers multiple dimensionality reduction approaches,
        each emphasizing different aspects of the data structure. Linear
        methods are generally easier to interpret and often provide a useful
        starting point for exploratory analysis. They are particularly
        effective when conformational variability follows approximately
        linear relationships.

        Nonlinear methods are better suited for datasets in which structural
        changes occur along curved manifolds or complex pathways. These
        approaches can uncover relationships that may be invisible to linear
        projections and are often valuable when studying highly flexible
        macromolecular assemblies.

        Different methods may produce different visualizations of the same
        dataset. Consequently, comparing several approaches can provide
        complementary insights into the organization of conformational
        variability.

        Reduced Dimensionality Selection

        One of the most important decisions is the number of dimensions to
        retain in the final representation. Two-dimensional projections are
        commonly used because they are easy to visualize and interpret.
        Three-dimensional representations may reveal additional structural
        complexity while remaining accessible for interactive exploration.

        Retaining too few dimensions may hide biologically relevant
        variability, whereas retaining too many dimensions can complicate
        interpretation. In exploratory studies, users often begin with two
        dimensions and subsequently evaluate whether additional dimensions
        provide meaningful new information.

        Interpretation of the Reduced Space

        Volumes located close together in the reduced space generally
        correspond to similar deformation patterns and therefore similar
        conformational states. Conversely, distant points typically indicate
        larger structural differences.

        Clusters may represent discrete biological states, while continuous
        trajectories can indicate gradual transitions between conformations.
        The biological significance of these patterns should always be
        assessed together with structural inspection and complementary
        experimental evidence.

        Projection and Reusability

        Some dimensionality reduction strategies generate transformation
        models that can later be applied to additional datasets. This allows
        newly obtained structures to be projected into an existing
        conformational landscape, facilitating comparisons across experiments,
        conditions, or processing campaigns.

        Such projections are particularly useful in longitudinal studies,
        comparative analyses, and iterative workflows where new data become
        available after the original analysis has been completed.

        Outputs and Their Interpretation

        The primary output is a reduced-coordinate representation of all
        analyzed volumes. Each volume is associated with a position in the
        reduced space that summarizes its deformation characteristics
        relative to the reference structure.

        Depending on the selected method, an additional transformation model
        may also be produced. This model can serve as a bridge between the
        original deformation descriptors and the reduced representation,
        enabling future projections and comparative analyses.

        Practical Recommendations

        For most biological applications, principal component analysis is an
        effective starting point because it provides a stable and easily
        interpretable description of dominant structural variability.
        Nonlinear methods become particularly valuable when the data suggest
        the presence of curved trajectories, branching pathways, or multiple
        interconnected conformational states.

        It is generally advisable to visualize the reduced coordinates,
        inspect possible clusters or trajectories, and compare the resulting
        organization with known biochemical, functional, or experimental
        information. Combining dimensionality reduction with structural
        visualization often yields the most biologically meaningful
        interpretation.

        Final Perspective

        Dimensionality reduction is a powerful tool for transforming complex
        deformation measurements into an interpretable representation of
        molecular flexibility. By revealing the dominant organization of
        conformational variability, this protocol helps researchers explore
        structural landscapes, identify biologically relevant states, and
        generate hypotheses regarding molecular function and dynamics.
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
