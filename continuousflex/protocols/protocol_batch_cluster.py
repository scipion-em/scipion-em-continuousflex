# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

from os.path import isfile
from pyworkflow.protocol.params import PointerParam, FileParam
from pwem.protocols import BatchProtocol
from pwem.objects import SetOfParticles, Volume, AtomStruct
from xmipp3.convert import writeSetOfParticles
from pwem.utils import runProgram
import pwem.emlib.metadata as md
import numpy as np


class FlexBatchProtNMACluster(BatchProtocol):
    """
    Creates a representative three-dimensional reconstruction and structural model from a cluster
    of particles associated with a Normal Mode Analysis exploration. The protocol summarizes a
    selected conformational population by generating both an average volumetric reconstruction and
    a corresponding molecular structure that represents the central deformation state of the
    cluster.

    AI Generated:

    NMA Cluster Reconstruction (FlexBatchProtNMACluster) - User Manual
        Overview

        The NMA Cluster Reconstruction protocol is designed to analyze groups of particles that
        belong to the same region of a Normal Mode Analysis conformational landscape. Its primary
        objective is to transform a cluster of related particle images into an interpretable
        structural representation that reflects the dominant characteristics of that population.

        In studies of molecular flexibility, dimensionality reduction and clustering are commonly
        used to identify groups of particles that share similar conformational properties. Once
        such a cluster has been identified, researchers often need a representative volume and a
        corresponding structural model that summarize the behavior of the selected population.
        This protocol provides those representative outputs.

        Inputs and General Workflow

        The protocol operates on a cluster derived from a previous Normal Mode Analysis workflow.
        The selected particles are gathered into a dedicated dataset while preserving the
        conformational information associated with each member of the cluster.

        The particle images are then combined to generate a three-dimensional reconstruction that
        represents the average structural state of the selected population. In parallel, the
        conformational information associated with the cluster is used to determine a central
        deformation state, allowing the generation of a representative structural model.

        Relationship Between Clustering and Conformational Landscapes

        Clusters within a Normal Mode Analysis landscape frequently correspond to regions occupied
        by related molecular conformations. Depending on the biological system, a cluster may
        represent a stable state, a transition intermediate, or a family of closely related
        structural arrangements.

        By focusing on a specific cluster, the protocol allows researchers to move from abstract
        coordinates in a reduced conformational space to physically interpretable structural
        representations. This connection is particularly valuable when investigating continuous
        motions, domain rearrangements, or large-scale conformational transitions.

        Three-Dimensional Reconstruction

        The reconstructed volume provides a consensus representation of the particle population
        contained within the cluster. Structural features consistently present across the selected
        particles tend to be reinforced, while random noise is reduced through the reconstruction
        process.

        For relatively homogeneous clusters, the resulting volume can provide a clear description
        of the underlying molecular state. When structural variability remains within the cluster,
        flexible regions may appear broadened or less sharply defined. Such behavior should be
        interpreted as evidence of residual heterogeneity rather than as a reconstruction artifact.

        Representative Structural Model

        In addition to the reconstructed volume, the protocol produces a structural model
        representing the central conformational state of the cluster. This model serves as a
        convenient structural reference for visualization, interpretation, and comparison with
        other regions of the conformational landscape.

        From a biological perspective, the representative structure can help identify the dominant
        motions associated with a cluster and provide insight into how conformational variability
        relates to molecular function. Comparisons between representative structures from different
        clusters may reveal transition pathways or alternative functional states.

        Outputs and Their Interpretation

        The protocol generates two complementary outputs. The first is a reconstructed volume that
        summarizes the experimental information contained within the particle cluster. The second
        is a representative molecular structure associated with the central conformational state of
        that cluster.

        Together, these outputs provide both an experimental and a structural description of the
        selected population, facilitating interpretation of conformational variability and
        biological function.

        Practical Recommendations

        The quality and interpretability of the results depend strongly on the coherence of the
        selected cluster. Clusters representing well-defined conformational states generally
        produce representative volumes and structures that are straightforward to interpret.

        When studying complex molecular motions, it is often beneficial to compare the outputs
        generated from multiple clusters. Such comparisons can reveal progressive structural
        changes across the conformational landscape and help identify biologically meaningful
        transitions.

        Visual inspection of both the reconstructed volume and the representative structure is
        recommended, particularly when clusters contain broad conformational variability or when
        multiple structural states may coexist within the same region of the landscape.

        Final Perspective

        For researchers investigating continuous molecular flexibility, this protocol provides a
        direct bridge between clustered particle populations and biologically interpretable
        structural representations. By generating both a consensus reconstruction and a
        representative conformational model, it helps transform abstract conformational clusters
        into tangible structural states that can be analyzed, compared, and communicated.
    """
    _label = 'nma cluster'
    
    def _defineParams(self, form):
        form.addHidden('inputNmaDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)
        
    #--------------------------- INSERT steps functions --------------------------------------------
        
    def _insertAllSteps(self):
        imagesMd = self._getExtraPath('images.xmd')
        outputVol = self._getExtraPath('reconstruction.vol')
        
        self._insertFunctionStep('convertInputStep', imagesMd)
        params = '-i %(imagesMd)s -o %(outputVol)s --fast' % locals()
        self._insertFunctionStep('reconstructStep', params)
        self._insertFunctionStep('centroidPdbStep')
        self._insertFunctionStep('createOutputStep', outputVol)
        
    #--------------------------- STEPS functions --------------------------------------------   
        
    def convertInputStep(self, imagesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        inputSet = self.inputNmaDimred.get().getInputParticles()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(inputSet)
        
        tmpSet = SetOfParticles(filename=self.sqliteFile.get())        
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()
        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)
        
        writeSetOfParticles(partSet, imagesMd)

        # Add the NMA displacement to clusters XMD files
        md_file_nma = md.MetaData(self.inputNmaDimred.get().getParticlesMD())
        md_file_org = md.MetaData(imagesMd)
        for objID in md_file_org:
            # if image name is the same, we add the nma displacement from nma to org
            id_org = md_file_org.getValue(md.MDL_ITEM_ID, objID)
            for j in md_file_nma:
                id_nma = md_file_nma.getValue(md.MDL_ITEM_ID, j)
                print(id_nma)
                if id_org == id_nma:
                    displacements = md_file_nma.getValue(md.MDL_NMA, j)
                    md_file_org.setValue(md.MDL_NMA, displacements, objID)
                    break
        md_file_org.write(imagesMd)


    def reconstructStep(self, params):
        runProgram('xmipp_reconstruct_fourier_accel', params)


    def centroidPdbStep(self):
        imagesMd = self._getExtraPath('images.xmd')
        md_file = md.MetaData(imagesMd)
        deformations = []
        for j in md_file:
            defor = md_file.getValue(md.MDL_NMA, j)
            if defor:
                deformations.append(defor)
        ampl = np.mean(np.array(deformations), axis= 0)
        print(self.getFnPDB())

        fnPDB, pseudo = self.getFnPDB()
        fnModeList = self.getFnModes()
        fnOutPDB = self._getExtraPath('centroid.pdb')
        params = " --pdb " + fnPDB
        params += " --nma " + fnModeList
        params += " -o " + fnOutPDB
        params += " --deformations " + ' '.join(str(i) for i in ampl)
        runProgram('xmipp_pdb_nma_deform', params)
    
    def createOutputStep(self, outputVol):
        vol = Volume()
        vol.setFileName(outputVol)
        vol.setSamplingRate(self.outputParticles.getSamplingRate())
        atm = AtomStruct()
        fnPDB, pseudo = self.getFnPDB()
        fnOutPDB = self._getExtraPath('centroid.pdb')
        atm.setPseudoAtoms(pseudo)
        atm.setFileName(fnOutPDB)
        atm.setVolume(vol)
        self._defineOutputs(centroidPDB=atm)
        self._defineOutputs(outputVol=vol)

    #--------------------------- Utility functions -----------------------------------------
    def getFnPDB(self):
        # This functions returns the path of the structure, false if is atomic, true if pseudoatomic
        path = self.inputNmaDimred.get().inputNMA.get()._getExtraPath('atoms.pdb')
        if isfile(path):
            return path, False
        else:
            path = self.inputNmaDimred.get().inputNMA.get()._getExtraPath('pseudoatoms.pdb')
            return path, True

    def getFnModes(self):
        return self.inputNmaDimred.get().getInputModes()

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary
    
    def _validate(self):
        errors = []
        return errors
    
    def _citations(self):
        return ['harastani2022continuousflex']
    
    def _methods(self):
        return []
    
