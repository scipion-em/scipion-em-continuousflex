# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Qiyu Jin
# *           Slavica Jonic                (slavica.jonic@upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Jan 2014
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
from pyworkflow.utils import isPower2, getListFromRangeString
from pyworkflow.utils.path import copyFile, cleanPath
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from pyworkflow.protocol.params import NumericRangeParam
import pwem as em
import pwem.emlib.metadata as md
from xmipp3.base import XmippMdRow
from xmipp3.convert import (writeSetOfParticles, xmippToLocation,
                            getImageLocation, createItemMatrix,
                            setXmippAttributes)
from .convert import modeToRow
from pwem import Domain
import multiprocessing

NMA_ALIGNMENT_WAV = 0
NMA_ALIGNMENT_PROJ = 1


class FlexProtAlignmentNMA(ProtAnalysis3D):
    """ Protocol for flexible angular alignment (HEMNMA). """
    _label = 'nma alignment'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', NumericRangeParam, expertLevel=params.LEVEL_ADVANCED,
                      label="Modes selection",
                      help='Select the normal modes that will be used for image analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for image analysis.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('inputParticles', params.PointerParam, label="Input particles",
                      pointerClass='SetOfParticles',
                      help='Select the set of particles that will be analyzed using normal modes.')

        form.addParam('copyDeformations', params.PathParam,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Precomputed results (for development)',
                      help='Only for tests during development. Enter a metadata file with precomputed elastic '
                           'and rigid-body alignment parameters and perform '
                           'all remaining steps using this file.')

        form.addSection(label='Combined elastic and rigid-body alignment')
        form.addParam('trustRegionScale', params.IntParam, default=1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Elastic-alignment trust region scale',
                      help='For elastic alignment, this parameter scales the initial '
                           'value of the trust region radius of CONDOR optimization. '
                           'The default value of 1 works in majority of cases. \n'
                           'This value should not be changed except by expert users. '
                           'Larger values (e.g., between 1 and 2) can be tried '
                           'for larger expected amplitudes of conformational change.')
        form.addParam('alignmentMethod', params.EnumParam, default=NMA_ALIGNMENT_WAV,
                      choices=['wavelets & splines', 'projection matching'],
                      label='Rigid-body alignment method',
                      help='For rigid-body alignment, Projection Matching method (faster) or '
                           'Wavelets and Splines method (more accurate) can be used. In the case of Wavelets and Splines, '
                           'the size of images should be a power of 2.')
        form.addParam('discreteAngularSampling', params.FloatParam, default=10,
                      label="Discrete angular sampling (deg)",
                      help='This is the angular step (in degrees) with which a library of reference projections '
                           'is computed for rigid-body alignment in Projection Matching and Wavelets methods. \n'
                           'This alignment is refined with Splines method when Wavelets and Splines alignment is chosen.')

        form.addParallelSection(threads=0, mpi=multiprocessing.cpu_count()//2-1)

        # --------------------------- INSERT steps functions --------------------------------------------

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """
        return self.inputModes.get().getPdb()

    def _insertAllSteps(self):
        atomsFn = self.getInputPdb().getFileName()
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('images.xmd')
        self.imgsFn_backup = self._getExtraPath('images_backup.xmd')
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

        if self.copyDeformations.empty():  # ONLY FOR DEBUGGING
            self._insertFunctionStep("performNmaStep", self.atomsFn, self.modesFn)
        else:
            self._insertFunctionStep('copyDeformationsStep', self.copyDeformations.get())

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self, atomsFn):
        # Write the modes metadata taking into account the selection
        self.writeModesMetaData()
        # Write a metadata with the normal modes information
        # to launch the nma alignment programs
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn)
        writeSetOfParticles(self.inputParticles.get(),self.imgsFn_backup)


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
            if not modeSelection or mode.getObjId() in modeSelection:
                row = XmippMdRow()
                modeToRow(mode, row)
                row.writeToMd(mdModes, mdModes.addObject())
        mdModes.write(self.modesFn)

    def copyDeformationsStep(self, deformationMd):
        copyFile(deformationMd, self.imgsFn)
        # We update the image paths based on image names (if computer on another computer or imported from another
        # project), and we need to set the item_id for each image
        inputSet = self.inputParticles.get()
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
        mdImgs.write(self.imgsFn)

    def performNmaStep(self, atomsFn, modesFn):
        sampling = self.inputParticles.get().getSamplingRate()
        discreteAngularSampling = self.discreteAngularSampling.get()
        trustRegionScale = self.trustRegionScale.get()
        odir = self._getTmpPath()
        imgFn = self.imgsFn

        args = "-i %(imgFn)s --pdb %(atomsFn)s --modes %(modesFn)s --sampling_rate %(sampling)f "
        args += "--discrAngStep %(discreteAngularSampling)f --odir %(odir)s --centerPDB "
        args += "--trustradius_scale %(trustRegionScale)d --resume "

        if self.getInputPdb().getPseudoAtoms():
            args += "--fixed_Gaussian "

        if self.alignmentMethod == NMA_ALIGNMENT_PROJ:
            args += "--projMatch "

        # runProgram("xmipp_nma_alignment", args % locals())
        self.runJob("xmipp_nma_alignment", args % locals(),
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

        cleanPath(self._getPath('nmaTodo.xmd'))

        inputSet = self.inputParticles.get()
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
        mdImgs.write(self.imgsFn)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        pdbPointer = self.inputModes.get()._pdbPointer

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self.imgsFn, sortByLabel=md.MDL_ITEM_ID))

        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(pdbPointer, partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        xdim = self.inputParticles.get().getDim()[0]
        if not isPower2(xdim):
            errors.append("Image dimension (%s) is not a power of two, consider resize them" % xdim)
        return errors

    def _citations(self):
        return ['harastani2022continuousflex','harastani2020hybrid','Jin2014']

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

    def _getLocalModesFn(self):
        modesFn = self.inputModes.get().getFileName()
        return self._getBasePath(modesFn)

    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
                           md.MDL_SHIFT_Y, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)
