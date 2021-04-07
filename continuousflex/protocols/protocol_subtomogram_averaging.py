# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
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

import os
from pwem.protocols import ProtAnalysis3D
from xmipp3.convert import writeSetOfVolumes, xmippToLocation, createItemMatrix, setXmippAttributes
import pwem as em
from pwem.objects import Volume
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params

WEDGE_MASK_NONE = 0
WEDGE_MASK_THRE = 1

REFERENCE_NONE = 0
REFERENCE_EXISTS = 1
REFERENCE_IMPORTED = 2


class FlexProtSubtomogramAveraging(ProtAnalysis3D):
    """ Protocol for subtomogram averaging. """
    _label = 'subtomogram averaging'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        form.addParam('StartingReference', params.EnumParam,
                      choices=['start from scratch', 'import a volume file and use it as reference',
                               'select a volume and use as reference'],
                      default=REFERENCE_NONE,
                      label='Starting reference', display=params.EnumParam.DISPLAY_COMBO,
                      help='Align from scratch of choose a template')
        form.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXISTS,
                      label="starting reference file",
                      help='Choose a starting reference from an external volume file')
        form.addParam('ReferenceImported', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_IMPORTED,
                      label="selected starting reference",
                      help='Choose an imported volume as a starting reference')
        form.addParam('NumOfIters', params.IntParam, default=10,
                      label='Number of iterations', help='How many times you want to iterate while performing'
                                                         ' subtomogram alignment and averaging.')
        form.addParam('dynamoTable', params.PathParam, allowsNull=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Import a Dynamo table',
                      help='import a Dynamo table that contains the STA parameters. This option will evaluate '
                           'the average and transform the Dynamo table to Scipion metadata format')
        form.addSection(label='Missing-wedge Compensation')
        form.addParam('WedgeMode', params.EnumParam,
                      choices=['Do not compensate', 'Compensate'],
                      default=WEDGE_MASK_THRE,
                      label='Wedge mode', display=params.EnumParam.DISPLAY_COMBO,
                      help='Choose to compensate for the missing wedge if aligning subtomograms.'
                           ' However, if you are working with previously aligned subtomograms, then its better not to.')
        form.addParam('tiltLow', params.IntParam, default=-60,
                      # expertLevel=params.LEVEL_ADVANCED,
                      condition='WedgeMode==%d' % WEDGE_MASK_THRE,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        form.addParam('tiltHigh', params.IntParam, default=60,
                      # expertLevel=params.LEVEL_ADVANCED,
                      condition='WedgeMode==%d' % WEDGE_MASK_THRE,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')
        form.addSection(label='Advanced parameters')
        form.addParam('frm_freq', params.FloatParam, default=0.25,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Maximum normalized pixel frequency',
                      help='The normalized frequency should be a number between -0.5 and 0.5 '
                           'The more it is, the bigger the search frequency is, the more time it demands, '
                           'keeping it as default is recommended')
        form.addParam('frm_maxshift', params.IntParam, default=10,
                      expertlevel=params.LEVEL_ADVANCED,
                      label='Maximum shift for rigid body search',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'Increase it if your target is far from the center of the volumes')
        form.addParallelSection(threads=0, mpi=5)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        self.outputVolume = self._getExtraPath('final_average.mrc')
        self.outputMD = self._getExtraPath('final_md.xmd')

        self._insertFunctionStep('convertInputStep')
        if self.dynamoTable.empty():
            self._insertFunctionStep('doAlignmentStep')
        else:
            self._insertFunctionStep('adaptDynamoStep', self.dynamoTable.get())
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes to align
        writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)

    def doAlignmentStep(self):
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()
        max_itr = self.NumOfIters.get()
        iter_result = self._getExtraPath('result.xmd')
        reference = None
        if self.StartingReference == REFERENCE_EXISTS:
            reference = self.ReferenceVolume.get()
        if self.StartingReference == REFERENCE_IMPORTED:
            reference = self.ReferenceImported.get().getFileName()

        print('tempdir is ', tempdir)
        print('imgFn is ', imgFn)
        print('frm_freq is ', frm_freq)
        print('frm_maxshift is ', frm_maxshift)
        print('max_itr is ', max_itr)
        print('iter_result is ', iter_result)
        print('reference is ', reference)

        # if the reference is None, then we got to make an initial reference:
        if reference is None:
            initialref = self._getExtraPath('initialref.mrc')
            mdImgs = md.MetaData(imgFn)
            counter = 0
            first = True
            tempVol = self._getExtraPath('temp.mrc')
            for objId in mdImgs:
                counter = counter + 1
                imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
                if counter == 1:
                    args = '-i %(imgPath)s -o %(tempVol)s --type vol' % locals()
                    self.runJob('xmipp_image_convert',args, numberOfMpi=1)
                else:
                    params = '-i %(imgPath)s --plus %(tempVol)s -o %(tempVol)s ' % locals()
                    self.runJob('xmipp_image_operate', params, numberOfMpi=1)
            params = '-i %(tempVol)s --divide %(counter)s -o %(initialref)s ' % locals()
            self.runJob('xmipp_image_operate', params, numberOfMpi=1)
            os.system("rm -f %(tempVol)s" % locals())
            reference = initialref

        for i in range(1, max_itr + 1):
            arg = 'params_itr_' + str(i) + '.xmd'
            md_itr = self._getExtraPath(arg)
            arg = 'average_itr_' + str(i) + '.mrc'
            avr_itr = self._getExtraPath(arg)
            args = "-i %(imgFn)s -o %(md_itr)s --odir %(tempdir)s --resume --ref %(reference)s" \
                   " --frm_parameters %(frm_freq)f %(frm_maxshift)d "

            if self.WedgeMode == WEDGE_MASK_THRE:
                tilt0 = self.tiltLow.get()
                tiltF = self.tiltHigh.get()
                # args += " %(tilt0)d %(tiltF)d "
                args += "--tilt_values %(tilt0)d %(tiltF)d "

            self.runJob("xmipp_volumeset_align", args % locals())

            # By now, the alignment is done, the averaging should take place

            mdImgs = md.MetaData(md_itr)
            counter = 0
            first = True

            for objId in mdImgs:
                counter = counter + 1

                imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
                rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
                tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
                psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

                x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
                y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
                z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

                flip = mdImgs.getValue(md.MDL_ANGLE_Y, objId)
                tempVol = self._getExtraPath('temp.mrc')
                extra = self._getExtraPath()

                if flip == 0:
                    if first:
                        print("THERE IS NO COMPENSATION FOR THE MISSING WEDGE")
                        first = False

                    params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                             ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

                else:
                    if first:
                        print("THERE IS A COMPENSATION FOR THE MISSING WEDGE")
                        first = False
                    # First got to rotate each volume 90 degrees about the y axis, align it, then rotate back and sum it
                    params = '-i %(imgPath)s -o %(tempVol)s --rotate_volume euler 0 90 0' % locals()
                    self.runJob('xmipp_transform_geometry', params, numberOfMpi=1)
                    params = '-i %(tempVol)s -o %(tempVol)s --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                             ' --shift %(x_shift)s %(y_shift)s %(z_shift)s ' % locals()

                self.runJob('xmipp_transform_geometry', params, numberOfMpi=1)

                if counter == 1:
                    os.system("cp %(tempVol)s %(avr_itr)s" % locals())

                else:
                    params = '-i %(tempVol)s --plus %(avr_itr)s -o %(avr_itr)s ' % locals()
                    self.runJob('xmipp_image_operate', params, numberOfMpi=1)

            params = '-i %(avr_itr)s --divide %(counter)s -o %(avr_itr)s ' % locals()
            self.runJob('xmipp_image_operate', params, numberOfMpi=1)
            os.system("rm -f %(tempVol)s" % locals())
            # Updating the reference then realigning:
            reference = avr_itr

        outputVolume = self.outputVolume
        outputMD = self.outputMD
        os.system("cp %(avr_itr)s %(outputVolume)s " % locals())
        os.system("cp %(md_itr)s %(outputMD)s " % locals())

        # Averaging is done


        inputSet = md.MetaData(self.imgsFn)
        mdImgs = md.MetaData(self.outputMD)

        # setting item_id (lost due to mpi usually)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fn = xmippToLocation(imgPath)
            # Conside the index is the id in the input set
            for objId2 in inputSet:
                NewImgPath = inputSet.getValue(md.MDL_IMAGE, objId2)
                if (NewImgPath == imgPath):
                    target_ID = inputSet.getValue(md.MDL_ITEM_ID, objId2)
                    break

            mdImgs.setValue(md.MDL_ITEM_ID, target_ID, objId)

        mdImgs.write(self.outputMD)

    def adaptDynamoStep(self, dynamoTable):
        volumes_in = self.imgsFn
        volume_out = self.outputVolume
        md_out = self.outputMD
        from continuousflex.protocols.utilities.dynamo import tbl2metadata
        tbl2metadata(dynamoTable, volumes_in, md_out)


        ### here:
        mdImgs = md.MetaData(md_out)
        counter = 0
        first = True

        for objId in mdImgs:
            counter = counter + 1

            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

            flip = mdImgs.getValue(md.MDL_ANGLE_Y, objId)
            tempVol = self._getExtraPath('temp.mrc')
            extra = self._getExtraPath()

            if flip == 0:
                if first:
                    print("Averaging based on Dynamo parameters")
                    first = False

                params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            else:
                if first:
                    print("THERE IS A COMPENSATION FOR THE MISSING WEDGE")
                    first = False
                # First got to rotate each volume 90 degrees about the y axis, align it, then rotate back and sum it
                params = '-i %(imgPath)s -o %(tempVol)s --rotate_volume euler 0 90 0' % locals()
                self.runJob('xmipp_transform_geometry', params, numberOfMpi=1)
                params = '-i %(tempVol)s -o %(tempVol)s --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s ' % locals()

            self.runJob('xmipp_transform_geometry', params, numberOfMpi=1)

            if counter == 1:
                os.system("cp %(tempVol)s %(volume_out)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(volume_out)s -o %(volume_out)s ' % locals()
                self.runJob('xmipp_image_operate', params, numberOfMpi=1)

        params = '-i %(volume_out)s --divide %(counter)s -o %(volume_out)s ' % locals()
        self.runJob('xmipp_image_operate', params, numberOfMpi=1)
        os.system("rm -f %(tempVol)s" % locals())
         # Averaging is done

        pass

    def createOutputStep(self):
        inputSet = self.inputVolumes.get()
        partSet = self._createSetOfVolumes()
        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(self.imgsFn, sortByLabel=md.MDL_ITEM_ID))
        outvolume = Volume()
        outvolume.setSamplingRate(inputSet.getSamplingRate())
        outvolume.setFileName(self.outputVolume)

        self._defineOutputs(outputParticles=partSet, outputvolume=outvolume)
        self._defineTransformRelation(self.inputVolumes, partSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return []

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

    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
                           md.MDL_SHIFT_Y, md.MDL_SHIFT_Z, md.MDL_MAXCC, md.MDL_ANGLE_Y)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)
