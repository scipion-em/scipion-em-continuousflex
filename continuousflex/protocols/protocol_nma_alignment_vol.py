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

from os.path import basename
import os
from pyworkflow.utils import getListFromRangeString
from pwem.protocols import ProtAnalysis3D
from xmipp3.convert import (writeSetOfVolumes, xmippToLocation, createItemMatrix,
                            setXmippAttributes, getImageLocation)
import pwem as em
import pwem.emlib.metadata as md
from xmipp3 import XmippMdRow
from pyworkflow.utils.path import copyFile, cleanPath
import pyworkflow.protocol.params as params
from pyworkflow.protocol.params import NumericRangeParam
from .convert import modeToRow, eulerAngles2matrix, matrix2eulerAngles
from pwem import Domain
import numpy as np
import multiprocessing

WEDGE_MASK_NONE = 0
WEDGE_MASK_THRE = 1


class FlexProtAlignmentNMAVol(ProtAnalysis3D):
    """
    Protocol for rigid-body and elastic alignment for volumes using Normal Mode Analysis (NMA). It enables the study of structural flexibility by fitting an atomic or pseudoatomic reference model into a collection of three-dimensional volumes while simultaneously estimating conformational changes and spatial alignment parameters. The protocol is particularly suited for cryo-electron tomography subtomograms and cryo-EM maps where structural variability is expected and quantitative characterization of continuous motions is required.

    AI Generated:

    NMA Volume Alignment (FlexProtAlignmentNMAVol) - User Manual
        Overview

        The NMA Volume Alignment protocol analyzes structural variability by combining elastic
        deformation modeling with rigid-body alignment. Its purpose is to determine how a
        reference structure must move and deform in order to best explain a collection of
        experimental three-dimensional volumes. Rather than treating each volume as an
        independent reconstruction, the protocol interprets them as different manifestations
        of a potentially continuous conformational landscape.

        For biological users, this approach is especially valuable when studying molecular
        machines, multi-domain proteins, membrane complexes, or other systems that exhibit
        flexibility. By describing each volume through a combination of normal mode amplitudes
        and spatial orientation parameters, the protocol provides a quantitative representation
        of conformational heterogeneity suitable for downstream analysis and visualization.

        Inputs and General Workflow

        The protocol requires a previously computed set of normal modes associated with an
        atomic or pseudoatomic structural model. These modes define the possible directions
        of motion available to the structure and provide a physically meaningful framework
        for describing conformational changes.

        In addition, the protocol requires one or more experimental volumes. Each volume is
        analyzed independently against the same structural model. During processing, the
        protocol searches simultaneously for the rigid-body transformation that places the
        structure into the volume and for the elastic deformation amplitudes that best
        reproduce the observed density.

        The resulting description captures both orientation and flexibility, allowing
        structural variability to be represented in a compact and biologically interpretable
        form.

        Selection of Normal Modes

        Biological interpretation depends strongly on the selected modes. Users may analyze
        all available modes or restrict the analysis to a subset of motions considered
        biologically relevant.

        In many applications, low-frequency collective modes provide the most meaningful
        description of large-scale conformational transitions. These modes often correspond
        to domain rearrangements, hinge motions, opening and closing events, or other
        functionally important structural changes.

        Restricting the analysis to biologically plausible modes can improve robustness and
        reduce the risk of fitting noise or reconstruction artefacts. Conversely, including
        a larger number of modes may be beneficial when the conformational landscape is
        expected to be complex.

        Missing-Wedge Compensation

        For subtomogram datasets, missing-wedge artefacts represent one of the most important
        sources of distortion. These artefacts arise from incomplete angular sampling during
        tomographic acquisition and can bias alignment and deformation estimates.

        The protocol provides an optional missing-wedge compensation strategy designed to
        account for this limitation during fitting. When enabled, the analysis incorporates
        information about the acquisition tilt range, improving the reliability of the
        recovered conformational parameters.

        For cryo-EM density maps or subtomograms that have already undergone appropriate
        missing-wedge correction, compensation may be unnecessary. Choosing the correct
        setting depends on the origin and preprocessing history of the data.

        Combined Elastic and Rigid-Body Alignment

        One of the defining characteristics of this protocol is the simultaneous treatment
        of structural deformation and spatial alignment. Traditional alignment approaches
        assume a rigid object and attempt only to determine orientation and translation.
        Such assumptions are often insufficient for flexible biological systems.

        Here, rigid-body positioning and conformational adaptation are optimized together.
        This allows the protocol to distinguish between genuine structural variability and
        simple differences in orientation. As a result, the recovered parameters provide a
        more realistic representation of the underlying molecular motions.

        The optimization procedure can be adjusted through advanced parameters that control
        the search behavior. For most biological applications, the default settings provide
        an appropriate balance between robustness and computational efficiency. Expert users
        studying highly flexible systems may choose to explore alternative settings when
        larger conformational amplitudes are expected.

        Interpretation of the Results

        The principal output is a set of volumes enriched with deformation and alignment
        information. Each analyzed volume receives a corresponding collection of rigid-body
        parameters together with amplitudes describing motion along the selected normal modes.

        Biologically, these amplitudes represent coordinates within a conformational space.
        Volumes with similar amplitudes correspond to related structural states, whereas
        larger differences indicate more substantial conformational changes. The resulting
        dataset can therefore be interpreted as a quantitative map of structural variability.

        Because the deformation parameters are expressed in terms of normal modes, the
        results remain connected to physically meaningful motions rather than arbitrary
        mathematical descriptors.

        Integration with Downstream Analysis

        The protocol is commonly used as a preparatory step for dimensionality reduction
        and conformational landscape exploration. Once deformation parameters have been
        estimated, they can be projected into lower-dimensional spaces where dominant
        motions and structural transitions become easier to visualize.

        Such analyses can reveal continuous trajectories, clusters of related states,
        transition pathways, and other features that help characterize the functional
        dynamics of the biological system under study.

        Practical Recommendations

        Before running the protocol, users should verify that the selected normal modes
        capture motions relevant to the biological question. Low-frequency collective modes
        are generally the most informative starting point.

        When analyzing subtomograms, enabling missing-wedge compensation is usually advisable
        unless a reliable correction procedure has already been applied. Accurate acquisition
        tilt limits should be provided whenever possible.

        The quality of the reference structure also plays a critical role. A model that
        adequately represents the overall architecture of the system will generally produce
        more meaningful deformation estimates than an incomplete or poorly matched reference.

        Final Perspective

        For many cryo-EM and cryo-electron tomography studies, understanding structural
        flexibility is as important as determining static structure. This protocol provides
        a framework for describing conformational variability directly from experimental
        volumes using physically interpretable normal modes. By combining elastic deformation
        analysis with rigid-body alignment, it enables researchers to characterize molecular
        motions, identify conformational states, and build quantitative models of structural
        dynamics across heterogeneous datasets.
    """
    _label = 'nma alignment vol'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', NumericRangeParam, expertLevel=params.LEVEL_ADVANCED,
                      label="Modes selection",
                      help='Select the normal modes that will be used for volume analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for image analysis.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select the set of volumes that will be analyzed using normal modes.')
        form.addParam('copyDeformations', params.PathParam,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Precomputed results (for development)',
                      help='Enter a metadata file with precomputed elastic  \n'
                           'and rigid-body alignment parameters to perform \n'
                           'remaining steps using this file.')
        form.addSection(label='Missing-wedge Compensation')
        form.addParam('WedgeMode', params.EnumParam,
                      choices=['Do not compensate', 'Compensate'],
                      default=WEDGE_MASK_THRE,
                      label='Wedge mode', display=params.EnumParam.DISPLAY_COMBO,
                      help='Choose to compensate for the missing wedge if the data is subtomograms.'
                           ' However, if you correct the missing wedge in advance, then choose not to compensate.'
                           ' You can also choose not to compensate if your data is not subtomograms but EM-maps.'
                           ' The missing wedge is assumed to be in the Y-axis direction.')
        form.addParam('tiltLow', params.IntParam, default=-60,
                      condition='WedgeMode==%d' % WEDGE_MASK_THRE,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        form.addParam('tiltHigh', params.IntParam, default=60,
                      condition='WedgeMode==%d' % WEDGE_MASK_THRE,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')

        form.addSection(label='Combined elastic and rigid-body alignment')
        form.addParam('trustRegionScale', params.FloatParam, default=1.0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Elastic alignment trust region scale ',
                      help='For elastic alignment, this parameter scales the initial '
                           'value of the trust region radius of CONDOR optimization. '
                           'The default value of 1 works in majority of cases. \n'
                           'This value should not be changed except by expert users. '
                           'Larger values (e.g., between 1 and 2) can be tried '
                           'for larger expected amplitudes of conformational change.')
        form.addHidden('rhoStartBase', params.FloatParam, default=250.0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='CONDOR optimiser parameter rhoStartBase',
                      help='rhoStartBase > 0  : (rhoStart = rhoStartBase*trustRegionScale) the lower the better,'
                           ' yet the slower')
        form.addHidden('rhoEndBase', params.FloatParam, default=50.0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='CONDOR optimiser parameter rhoEndBase ',
                      help='rhoEndBase > 250  : (rhoEnd = rhoEndBase*trustRegionScale) no specific rule, '
                           'however it is better to keep it < 1000 if set very high we risk distortions')
        form.addHidden('niter', params.IntParam, default=10000,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='CONDOR optimiser parameter niter',
                      help='niter should be big enough to guarantee that the search converges to the '
                           'right set of nma deformation amplitudes')
        form.addParam('frm_freq', params.FloatParam, default=0.25,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Maximum cross correlation frequency',
                      help='The normalized frequency should be between 0 and 0.5 \n'
                           'The larger it is, the bigger the search frequency is, the more time it demands. '
                           'Keeping it as default is recommended.')
        form.addParam('frm_maxshift', params.IntParam, default=10,
                      expertlevel=params.LEVEL_ADVANCED,
                      label='Maximum shift for rigid body alignment (in pixels)',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'It represents the maximum distance searched in x, y and z directions. Keep as default'
                           ' if your target is near the center in your subtomograms')
        form.addParallelSection(threads=0, mpi=multiprocessing.cpu_count()//2-1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """
        return self.inputModes.get().getPdb()

    def _insertAllSteps(self):
        atomsFn = self.getInputPdb().getFileName()
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        self.imgsFn_backup = self._getExtraPath('volumes_backup.xmd')
        self.modesFn = self._getExtraPath('modes.xmd')
        self.structureEM = self.inputModes.get().getPdb().getPseudoAtoms()
        if self.structureEM:
            self.atomsFn = self._getExtraPath(basename(atomsFn))
            copyFile(atomsFn, self.atomsFn)
        else:
            pdb_name = os.path.dirname(self.inputModes.get().getFileName()) + '/atoms.pdb'
            self.atomsFn = self._getExtraPath(basename(pdb_name))
            copyFile(pdb_name, self.atomsFn)

        self._insertFunctionStep('convertInputStep', atomsFn)

        if self.copyDeformations.empty():
            self._insertFunctionStep("performNmaStep", self.atomsFn, self.modesFn)
        else:  # SERVES FOR DEBUGGING AND COMPUTING ON CLUSTERS
            self._insertFunctionStep('copyDeformationsStep', self.copyDeformations.get())

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self, atomsFn):
        # Write the modes metadata taking into account the selection
        self.writeModesMetaData()
        # Write a metadata with the normal modes information
        # to launch the nma alignment programs
        writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn_backup)

    def writeModesMetaData(self):
        """ Iterate over the input SetOfNormalModes and write
        the proper Xmipp metadata.
        Take into account a possible selection of modes
        """

        if self.modeList.empty():
            modeSelection = []
        else:
            modeSelection = getListFromRangeString(self.modeList.get())

        mdModes = md.MetaData()

        inputModes = self.inputModes.get()
        for mode in inputModes:
            # If there is a mode selection, only
            # take into account those selected
            if not modeSelection or mode.getObjId() in modeSelection:
                row = XmippMdRow()
                modeToRow(mode, row)
                row.writeToMd(mdModes, mdModes.addObject())
        mdModes.write(self.modesFn)

    def copyDeformationsStep(self, deformationMd):
        copyFile(deformationMd, self.imgsFn)
        # We update the volume paths based on volume names (if computed on another computer or imported from another
        # project), and we need to set the item_id for each volume
        inputSet = self.inputVolumes.get()
        mdImgs = md.MetaData(self.imgsFn)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fn = xmippToLocation(imgPath)
            if(index): # case the input is a stack
                # Conside the index is the id in the input set
                particle = inputSet[index]
            else: # input is not a stack
                # convert the inputSet to metadata:
                mdtemp = md.MetaData(self.imgsFn_backup)
                # Loop and find the index based on the basename:
                bn_retrieved = basename(imgPath)
                for searched_index in mdtemp:
                    imgPath_temp = mdtemp.getValue(md.MDL_IMAGE,searched_index)
                    bn_searched = basename(imgPath_temp)
                    if bn_searched == bn_retrieved:
                        index = searched_index
                        particle = inputSet[index]
                        break
            mdImgs.setValue(md.MDL_IMAGE, getImageLocation(particle), objId)
            mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        mdImgs.sort(md.MDL_ITEM_ID)
        mdImgs.write(self.imgsFn)

        # if the volumes were aligned with angle_y=90 degrees, then rotate by 90 and inverse, then set angle y to 0
        mdImgs = md.MetaData(self.imgsFn)

        flag = None
        try:
            flag = mdImgs.getValue(md.MDL_ANGLE_Y, 1)
        except:
            pass

        if flag == 90:
            mdImgs = md.MetaData(self.imgsFn)
            for objId in mdImgs:
                rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
                tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
                psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)
                x = mdImgs.getValue(md.MDL_SHIFT_X, objId)
                y = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
                z = mdImgs.getValue(md.MDL_SHIFT_Z, objId)
                T = eulerAngles2matrix(rot, tilt, psi, x, y, z)
                # Rotate 90 degrees (compensation for missing wedge)
                T0 = eulerAngles2matrix(0, 90, 0, 0, 0, 0)
                T = np.linalg.inv(np.matmul(T, T0))
                rot, tilt, psi, x, y, z = matrix2eulerAngles(T)
                mdImgs.setValue(md.MDL_ANGLE_ROT, rot, objId)
                mdImgs.setValue(md.MDL_ANGLE_TILT, tilt, objId)
                mdImgs.setValue(md.MDL_ANGLE_PSI, psi, objId)
                mdImgs.setValue(md.MDL_SHIFT_X, x, objId)
                mdImgs.setValue(md.MDL_SHIFT_Y, y, objId)
                mdImgs.setValue(md.MDL_SHIFT_Z, z, objId)
                mdImgs.setValue(md.MDL_ANGLE_Y, 0.0, objId)
            mdImgs.write(self.imgsFn)



    def performNmaStep(self, atomsFn, modesFn):
        sampling = self.inputVolumes.get().getSamplingRate()
        trustRegionScale = self.trustRegionScale.get()
        odir = self._getTmpPath()
        imgFn = self.imgsFn
        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()
        rhoStartBase = self.rhoStartBase.get()
        rhoEndBase = self.rhoEndBase.get()
        niter = self.niter.get()
        rhoStartBase = 250.0
        rhoEndBase = 50.0
        niter = 10000

        args = "-i %(imgFn)s --pdb %(atomsFn)s --modes %(modesFn)s --sampling_rate %(sampling)f "
        args += "--odir %(odir)s --centerPDB "
        args += "--trustradius_scale %(trustRegionScale)d --resume "

        if self.getInputPdb().getPseudoAtoms():
            args += "--fixed_Gaussian "

        args += "--alignVolumes %(frm_freq)f %(frm_maxshift)d "

        args += "--condor_params %(rhoStartBase)f %(rhoEndBase)f %(niter)d "

        if self.WedgeMode == WEDGE_MASK_THRE:
            tilt0 = self.tiltLow.get()
            tiltF = self.tiltHigh.get()
            args += "--tilt_values %(tilt0)d %(tiltF)d "

        self.runJob("xmipp_nma_alignment_vol", args % locals(),
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

        cleanPath(self._getPath('nmaTodo.xmd'))

        inputSet = self.inputVolumes.get()
        mdImgs = md.MetaData(self.imgsFn)

        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fn = xmippToLocation(imgPath)
            if(index): # case the input is a stack
                # Conside the index is the id in the input set
                particle = inputSet[index]
            else: # input is not a stack
                # convert the inputSet to metadata:
                mdtemp = md.MetaData(self.imgsFn_backup)
                # Loop and find the index based on the basename:
                bn_retrieved = basename(imgPath)
                for searched_index in mdtemp:
                    imgPath_temp = mdtemp.getValue(md.MDL_IMAGE,searched_index)
                    bn_searched = basename(imgPath_temp)
                    if bn_searched == bn_retrieved:
                        index = searched_index
                        particle = inputSet[index]
                        break
            mdImgs.setValue(md.MDL_IMAGE, getImageLocation(particle), objId)
            mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        mdImgs.sort(md.MDL_ITEM_ID)
        mdImgs.write(self.imgsFn)

        # if WedgeMode was Mask, then update the metadata angles and shifts to be in the same convention of the
        # ground truth (rotate 90 degrees) then set angle y to 0
        if self.WedgeMode == WEDGE_MASK_THRE:
            mdImgs = md.MetaData(self.imgsFn)
            for objId in mdImgs:
                rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
                tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
                psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)
                x = mdImgs.getValue(md.MDL_SHIFT_X, objId)
                y = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
                z = mdImgs.getValue(md.MDL_SHIFT_Z, objId)
                T = eulerAngles2matrix(rot, tilt, psi, x, y, z)
                # Rotate 90 degrees (compensation for missing wedge)
                T0 = eulerAngles2matrix(0, 90, 0, 0, 0, 0)
                T = np.linalg.inv(np.matmul(T, T0))
                rot, tilt, psi, x, y, z = matrix2eulerAngles(T)
                mdImgs.setValue(md.MDL_ANGLE_ROT, rot, objId)
                mdImgs.setValue(md.MDL_ANGLE_TILT, tilt, objId)
                mdImgs.setValue(md.MDL_ANGLE_PSI, psi, objId)
                mdImgs.setValue(md.MDL_SHIFT_X, x, objId)
                mdImgs.setValue(md.MDL_SHIFT_Y, y, objId)
                mdImgs.setValue(md.MDL_SHIFT_Z, z, objId)
                mdImgs.setValue(md.MDL_ANGLE_Y, 0.0, objId)
            mdImgs.write(self.imgsFn)

        cleanPath(self._getExtraPath('copy.xmd'))

    def createOutputStep(self):
        inputSet = self.inputVolumes.get()
        # partSet = self._createSetOfParticles()
        partSet = self._createSetOfVolumes()
        pdbPointer = self.inputModes.get()._pdbPointer

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self.imgsFn, sortByLabel=md.MDL_ITEM_ID))

        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(pdbPointer, partSet)
        self._defineTransformRelation(self.inputVolumes, partSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2021hemnma','harastani2022continuousflex']

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
                           md.MDL_SHIFT_Y, md.MDL_SHIFT_Z, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST, md.MDL_MAXCC)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)
