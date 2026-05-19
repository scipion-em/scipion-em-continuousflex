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

from os.path import isfile
from pyworkflow.protocol.params import PointerParam, FileParam
from pwem.protocols import BatchProtocol
from pwem.objects import Volume, SetOfVolumes, AtomStruct
from xmipp3.convert import writeSetOfVolumes
import pwem.emlib.metadata as md
import os
from pwem.utils import runProgram
import numpy as np


class FlexBatchProtNMAClusterVol(BatchProtocol):
    """
    Generates a representative structural and volumetric description of a cluster of
    conformations obtained from normal mode analysis. The protocol summarizes the
    structural variability present within a selected group of volumes by producing both
    an average density map and a representative molecular model corresponding to the
    central tendency of the cluster.

    AI Generated:

    NMA Volume Cluster (FlexBatchProtNMAClusterVol) - User Manual
        Overview

        The NMA Volume Cluster protocol is designed to analyze a subset of volumes that
        belong to the same conformational cluster after dimensionality reduction and
        normal mode analysis. Its main objective is to provide a biologically meaningful
        representation of the conformational state described by the cluster by combining
        information from all member volumes into a single consensus result.

        In studies of molecular flexibility, clusters often represent groups of
        structures sharing similar conformations. Rather than inspecting every volume
        individually, researchers can use this protocol to obtain a compact summary of
        the structural characteristics of an entire conformational population.

        Inputs and General Workflow

        The protocol operates on a cluster extracted from a previous normal mode
        analysis workflow. The selected volumes are gathered together and their
        associated conformational descriptors are preserved so that both structural
        and dynamical information remain available throughout the analysis.

        The workflow produces two complementary outputs. First, it generates an average
        volume that represents the overall density distribution of the cluster.
        Second, it creates a representative structural model corresponding to the
        average conformational state observed among all cluster members.

        Cluster Averaging

        A central component of the protocol is the generation of a consensus volume.
        All volumes belonging to the cluster contribute to this result, allowing the
        protocol to emphasize structural features that are consistently present across
        the population while reducing the influence of noise and individual variations.

        From a biological perspective, the resulting average volume can be interpreted
        as the characteristic density map of the conformational state represented by
        the cluster. This is particularly useful when exploring continuous molecular
        motions where individual structures may differ slightly but still belong to
        the same functional state.

        Interpretation of Conformational Variability

        In addition to density information, the protocol considers the conformational
        descriptors associated with the normal mode analysis. These descriptors
        characterize the position of each structure within the conformational landscape.

        By combining information from all members of the cluster, the protocol derives
        a representative conformational state that reflects the average behavior of the
        population. This provides a useful reference for understanding the dominant
        structural characteristics of the cluster and facilitates comparison with other
        conformational states identified during the analysis.

        Representative Structural Model

        The protocol generates a molecular structure corresponding to the centroid of
        the cluster. Biologically, this model can be interpreted as the structure that
        best represents the average conformation sampled by the cluster population.

        This centroid model is particularly valuable when visualizing molecular motions,
        comparing conformational states, preparing figures, or selecting representative
        structures for downstream analyses. Because it reflects an average state rather
        than a single observation, it often provides a clearer description of the
        conformational ensemble.

        Outputs and Their Interpretation

        The protocol produces an average volume representing the consensus density of
        the cluster and a representative atomic or pseudoatomic structure describing
        the centroid conformation. These outputs complement each other by providing
        both volumetric and structural views of the same conformational state.

        The average volume can be used for visualization, comparison with experimental
        maps, or subsequent image-processing tasks. The centroid structure can be used
        for structural interpretation, animation of molecular motions, fitting
        procedures, or integration with additional modeling workflows.

        Practical Recommendations

        This protocol is most informative when applied to clusters that represent
        coherent conformational populations. Well-defined clusters generally produce
        representative averages that preserve biologically meaningful structural
        features. If a cluster contains highly heterogeneous conformations, the
        resulting average may become less representative of any individual state.

        When comparing multiple clusters, examining both the centroid structures and
        the corresponding average volumes can provide valuable insight into the nature
        of the conformational transitions captured by the normal mode analysis.

        Final Perspective

        For researchers studying molecular flexibility, this protocol serves as a bridge
        between large collections of conformationally related volumes and an
        interpretable biological description of the underlying structural state. By
        generating a consensus density map together with a representative structural
        model, it enables efficient exploration and communication of conformational
        variability within complex molecular systems.
    """
    _label = 'nma vol cluster'

    def _defineParams(self, form):
        form.addHidden('inputNmaDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)

    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        outputVol = self._getExtraPath('average.vol')

        self._insertFunctionStep('convertInputStep', volumesMd)
        self._insertFunctionStep('averagingStep')
        self._insertFunctionStep('centroidPdbStep')
        self._insertFunctionStep('createOutputStep', outputVol)

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, volumesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        inputSet = self.inputNmaDimred.get().getInputParticles()
        partSet = self._createSetOfVolumes()
        partSet.copyInfo(inputSet)
        tmpSet = SetOfVolumes(filename=self.sqliteFile.get())
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()

        self._defineOutputs(OutputVolumes=partSet)
        self._defineTransformRelation(inputSet, partSet)
        writeSetOfVolumes(partSet, volumesMd)

        # Add the NMA displacement to clusters XMD files
        md_file_nma = md.MetaData(self.inputNmaDimred.get().getParticlesMD())
        md_file_org = md.MetaData(volumesMd)
        for objID in md_file_org:
            # if image name is the same, we add the nma displacement from nma to org
            id_org = md_file_org.getValue(md.MDL_ITEM_ID, objID)
            for j in md_file_nma:
                id_nma = md_file_nma.getValue(md.MDL_ITEM_ID, j)
                if id_org == id_nma:
                    displacements = md_file_nma.getValue(md.MDL_NMA, j)
                    md_file_org.setValue(md.MDL_NMA, displacements, objID)
                    break
        md_file_org.write(volumesMd)



    def averagingStep(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        mdVols = md.MetaData(volumesMd)

        counter = 0
        for objId in mdVols:
            counter = counter + 1
            imgPath = mdVols.getValue(md.MDL_IMAGE, objId)

            rot = mdVols.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdVols.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdVols.getValue(md.MDL_ANGLE_PSI, objId)
            x_shift = mdVols.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdVols.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdVols.getValue(md.MDL_SHIFT_Z, objId)

            outputVol = self._getExtraPath('average.vol')
            tempVol = self._getExtraPath('temp.vol')
            extra = self._getExtraPath()

            params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                     ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()
            runProgram('xmipp_transform_geometry', params)

            if counter == 1:
                os.system("mv %(tempVol)s %(outputVol)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(outputVol)s -o %(outputVol)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(outputVol)s --divide %(counter)s -o %(outputVol)s ' % locals()
        runProgram('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())


    def centroidPdbStep(self):
        volumesMd = self._getExtraPath('volumes.xmd')
        md_file = md.MetaData(volumesMd)
        deformations = []
        for j in md_file:
            deformations.append(md_file.getValue(md.MDL_NMA, j))
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
        vol.setSamplingRate(self.OutputVolumes.getSamplingRate())
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
        return self.inputNmaDimred.get().inputNMA.get()._getExtraPath('modes.xmd')

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
