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
    """ Protocol for rigid-body and elastic alignment for volumes using NMA. This protocol is the code module of HEMNMA-3D.
     It will take as input a set of normal modes calculated for an input atomic or pseudoatomic structure, and a set of volumes (subtomograms) to analyze.
     It fits the input structure using its modes (a subset of the modes need to be selected) into each one of the input volumes while simultaneously looking for rigid-body alignment, with
     compensation for missing wedge artefacts.
     The result of this protocol are rigid-body and elastic parameters for each input volume.
     Those results will be fed for a dimensionality reduction method (nma dimred vol) for further processing.
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
