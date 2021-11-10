# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
# *             Rémi Vuillemot               (remi.vuillemot@upmc.fr)
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

from os.path import basename, exists
import os
from pyworkflow.utils import getListFromRangeString
from pwem.protocols import ProtAnalysis3D
from xmipp3.convert import (writeSetOfVolumes, xmippToLocation, createItemMatrix,
                            setXmippAttributes, setOfParticlesToMd, getImageLocation)
import pwem as em
import pwem.emlib.metadata as md
from xmipp3 import XmippMdRow
from pyworkflow.utils.path import copyFile, cleanPath, makePath
import pyworkflow.protocol.params as params
from pyworkflow.protocol.params import NumericRangeParam
from .convert import modeToRow
from pwem.convert.atom_struct import cifToPdb
from pyworkflow.utils import replaceBaseExt
from pwem.utils import runProgram
from pwem import Domain


class FlexProtNMOF(ProtAnalysis3D):
    """ Protocol for fitting using normal mode analysis and 3D dense optical flow. """
    _label = 'nmof'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('NMA', params.BooleanParam, label='Use normal modes and optical flows?',
                      default=True)
        group = form.addGroup('Input normal modes',
                               condition='NMA')
        group.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes", allowsNull=True,
                      help='Set of modes computed by normal mode analysis.')
        group.addParam('modeList', NumericRangeParam,
                      label="Modes selection (optional)",
                      help='Select the normal modes that will be used for volume analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for image analysis.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        group = form.addGroup('Input PDB (using optical flow fitting only)',
                              condition='NMA is False')
        group.addParam('pdb', params.PointerParam, label='input (pseudo)atomic structure',
                       pointerClass='AtomStruct', allowsNull=True,
                       help='The input structure can be an atomic model '
                            '(true PDB) or a pseudoatomic model\n'
                            '(an EM volume converted into pseudoatoms)')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select a volume or a set of volumes that will be fitted using the method')
        form.addParam('iterations', params.IntParam, default=1,
                      label='number of iterations',
                      help = 'To do')
        form.addParam('do_rigidbody', params.BooleanParam, default=True,
                      label='Perform rigid-body alignment on the volume(s)?')
        group = form.addGroup('rigid-body alignment settings (Fast rotational matching)',
                              condition='do_rigidbody')
        group.addParam('frm_freq', params.FloatParam, default=0.25,
                      label='Maximum cross correlation frequency',
                      help='The normalized frequency should be between 0 and 0.5 '
                           'The more it is, the bigger the search frequency is, the more time it demands, '
                           'keeping it as default is recommended.')
        group.addParam('frm_maxshift', params.IntParam, default=10,
                      label='Maximum shift for rigid body alignment (in pixels)',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'It represents the maximum distance searched in x,y and z directions. Keep as default'
                           ' if your target is near the center in your subtomograms')

        form.addParallelSection(threads=0, mpi=5)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        self.imgsFn_backup = self._getExtraPath('volumes_backup.xmd')
        self.modesFn = self._getExtraPath('modes.xmd')
        # Copy the input PDB to self.atomsFn
        if(self.NMA.get()):
            atomsFn = self.getInputPdb().getFileName()
            self.structureEM = self.inputModes.get().getPdb().getPseudoAtoms()
            if self.structureEM:
                self.atomsFn = self._getExtraPath(basename(atomsFn))
                copyFile(atomsFn, self.atomsFn)
            else:
                pdb_name = os.path.dirname(self.inputModes.get().getFileName()) + '/atoms.pdb'
                self.atomsFn = self._getExtraPath(basename(pdb_name))
                copyFile(pdb_name, self.atomsFn)
        else:
            atomsFn = self.pdb.get().getFileName()
            self.atomsFn = self._getExtraPath(basename(atomsFn))
            copyFile(atomsFn, self.atomsFn)

        # Write a metadata file input volumes and maybe modes
        self._insertFunctionStep('convertInputStep')

        # Align volumes if needed
        if(self.do_rigidbody):
            self._insertFunctionStep('alignInputVolumes')

        self._insertFunctionStep("performNMOF")
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        if(self.NMA.get()):
            # Write the modes metadata taking into account the selection
            self.writeSelectedModesMetaData()
        # Write metadata file with input volumes
        try:
            # If we have a set of volumes
            writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        except:
            # If one volume
            mdF = md.MetaData()
            mdF.setValue(md.MDL_IMAGE, self.inputVolumes.get().getFileName(), mdF.addObject())
            mdF.write(self.imgsFn)


    def writeSelectedModesMetaData(self):
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

    def alignInputVolumes(self):
        # Generate a volume from the input PDB:
        atomFn = self.atomsFn
        reference = self._getExtraPath('ref')
        sampling = self.inputVolumes.get().getSamplingRate()
        size = self.inputVolumes.get().getXDim()
        args = "-i %(atomFn)s -o %(reference)s --sampling %(sampling)s --size %(size)s" % locals()
        # print('xmipp_volume_from_pdb ', args)
        runProgram('xmipp_volume_from_pdb', args)
        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()
        tempdir = self._getTmpPath('')
        # Perform the alignment using MPI
        imgFn = self.imgsFn
        imgFnAligned = self._getExtraPath('aligned.xmd')
        # The stupid xmipp program adds .vol by default
        reference = self._getExtraPath('ref.vol')
        args = "-i %(imgFn)s -o %(imgFnAligned)s --odir %(tempdir)s --resume --ref %(reference)s" \
               " --frm_parameters %(frm_freq)f %(frm_maxshift)d " % locals()
        # print('xmipp_volumeset_align ', args)
        # This is needed to run MPI
        self.runJob("xmipp_volumeset_align", args,
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        # apply the alignment
        alignedPath = self._getExtraPath('alinged/')
        makePath(alignedPath)
        mdImgs = md.MetaData(imgFnAligned)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)
            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

            new_imgPath = alignedPath+basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)

            params = '-i %(imgPath)s -o %(new_imgPath)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                     ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            runProgram('xmipp_transform_geometry', params)
        mdImgs.write(imgFnAligned)
        self.imgsFn = imgFnAligned


    def performNMOF(self):
        reference = self._getExtraPath('ref.vol')
        if(exists(reference)): # if rigid-body alignment was done
            pass
        else:
            atomFn = self.atomsFn
            reference = self._getExtraPath('ref')
            sampling = self.inputVolumes.get().getSamplingRate()
            size = self.inputVolumes.get().getXDim()
            args = "-i %(atomFn)s -o %(reference)s --sampling %(sampling)s --size %(size)s" % locals()
            # print('xmipp_volume_from_pdb ', args)
            runProgram('xmipp_volume_from_pdb', args)
            reference = self._getExtraPath('ref.vol')


            

        # args = "-i %(imgFn)s --pdb %(atomsFn)s --modes %(modesFn)s --sampling_rate %(sampling)f "
        # args += "--odir %(odir)s --centerPDB "
        # args += "--trustradius_scale %(trustRegionScale)d --resume "
        #
        # if self.getInputPdb().getPseudoAtoms():
        #     args += "--fixed_Gaussian "
        #
        # args += "--alignVolumes %(frm_freq)f %(frm_maxshift)d "
        #
        # args += "--condor_params %(rhoStartBase)f %(rhoEndBase)f %(niter)d "

        # if self.WedgeMode == WEDGE_MASK_THRE:
        #     tilt0 = self.tiltLow.get()
        #     tiltF = self.tiltHigh.get()
        #     args += "--tilt_values %(tilt0)d %(tiltF)d "

        # print(args % locals())
        # runProgram("xmipp_nma_alignment_vol", args % locals())
        # self.runJob("xmipp_nma_alignment_vol", args % locals(),
        #             env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

        # cleanPath(self._getPath('nmaTodo.xmd'))
        #
        # inputSet = self.inputVolumes.get()
        # mdImgs = md.MetaData(self.imgsFn)
        #
        # for objId in mdImgs:
        #     imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
        #     index, fn = xmippToLocation(imgPath)
        #     if(index): # case the input is a stack
        #         # Conside the index is the id in the input set
        #         particle = inputSet[index]
        #     else: # input is not a stack
        #         # convert the inputSet to metadata:
        #         mdtemp = md.MetaData(self.imgsFn_backup)
        #         # Loop and find the index based on the basename:
        #         bn_retrieved = basename(imgPath)
        #         for searched_index in mdtemp:
        #             imgPath_temp = mdtemp.getValue(md.MDL_IMAGE,searched_index)
        #             bn_searched = basename(imgPath_temp)
        #             if bn_searched == bn_retrieved:
        #                 index = searched_index
        #                 particle = inputSet[index]
        #                 break
        #     mdImgs.setValue(md.MDL_IMAGE, getImageLocation(particle), objId)
        #     mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        # mdImgs.sort(md.MDL_ITEM_ID)
        # mdImgs.write(self.imgsFn)
        #
        # mdImgs.write(self.imgsFn)
        # cleanPath(self._getExtraPath('copy.xmd'))

    def createOutputStep(self):
        pass
        # inputSet = self.inputVolumes.get()
        # # partSet = self._createSetOfParticles()
        # partSet = self._createSetOfVolumes()
        # pdbPointer = self.inputModes.get()._pdbPointer
        #
        # partSet.copyInfo(inputSet)
        # partSet.setAlignmentProj()
        # partSet.copyItems(inputSet,
        #                   updateItemCallback=self._updateParticle,
        #                   itemDataIterator=md.iterRows(self.imgsFn, sortByLabel=md.MDL_ITEM_ID))
        #
        # self._defineOutputs(outputParticles=partSet)
        # self._defineSourceRelation(pdbPointer, partSet)
        # self._defineTransformRelation(self.inputVolumes, partSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2020hybrid','Jonic2005', 'Sorzano2004b', 'Jin2014']

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
                           md.MDL_SHIFT_Y, md.MDL_SHIFT_Z, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST, md.MDL_MAXCC,
                           md.MDL_ANGLE_Y)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """

        return self.inputModes.get().getPdb()
