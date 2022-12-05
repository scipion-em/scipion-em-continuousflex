# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
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
from xmipp3.convert import writeSetOfVolumes, xmippToLocation
from pwem.objects import Volume
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pwem.utils import runProgram
from pwem import Domain
from .convert import eulerAngles2matrix, matrix2eulerAngles
import numpy as np
import multiprocessing

WEDGE_MASK_NONE = 0
WEDGE_MASK_THRE = 1

REFERENCE_NONE = 0
REFERENCE_EXISTS = 1
REFERENCE_IMPORTED = 2

PERFORM_STA = 0
COPY_STA = 1

IMPORT_XMIPP_MD = 0
IMPORT_DYNAMO_TBL = 1
IMPORT_TOMBOX_MTV = 2

class FlexProtSubtomogramAveraging(ProtAnalysis3D):
    """ Protocol for subtomogram averaging. This protocol has two modes of operation.
     the first is to perform subtomogram averaging using Fast Rotational Matching.
     The second mode is to import a previously performed alignment using this protocol, Dynamo, or Artiatomi.
     If an alignment is imported, the rigid-body parameters will be used to re-create the average structure.
     """

    _label = 'subtomogram averaging'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='')
        group = form.addGroup('Settings')
        # form.addSection(label='Input')
        group.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        group.addParam('StA_choice', params.EnumParam,
                      choices=['Perform StA using Fast Rotational Matching (FRM)',
                               'Import parameters of a previously performed StA'],
                      default=PERFORM_STA,
                      label='Choose what processes you want to perform:', display=params.EnumParam.DISPLAY_COMBO,
                       help='If you choose to "Perform StA" using FRM you have to set the parameters in the last tab.'
                            ' Otherwise, if you import a pervious StA, the remaining parameters are not used')
        group = form.addGroup('Importing a previous StA protocol',
                              condition='StA_choice==%d'% COPY_STA)
        group.addParam('import_choice', params.EnumParam,
                       label='From which software?',
                       choices=['Import Scipion/Xmipp metadata',
                                'Import Dynamo table',
                                'Import TOM-ToolBox motive list'],
                       default=IMPORT_XMIPP_MD,
                       help='You have to provide a pervious table of rigid-body alignment parameters in one of the list'
                            'of supported formats. The software will evaluate the average based on the provided file. '
                            'If the average is correct (corresponds to what you have before) then the alignment went correctly,'
                            ' and in this case you can proceed in further processing (refinement and heterogeneity analysis).'
                       )
        group.addParam('xmippMD', params.PathParam, allowsNull=True,
                       condition='import_choice==%d' %IMPORT_XMIPP_MD,
                      label='Import Scipion/Xmipp metadata file',
                      help='import a the metadata file that contains the StA parameters. This option will evaluate '
                           'the average and allows you to perform post-StA processes (refinement and heterogeneity analysis).')
        group.addParam('dynamoTable', params.PathParam, allowsNull=True,
                       condition='import_choice==%d' % IMPORT_DYNAMO_TBL,
                       label='Import a Dynamo table [Beta]',
                      help='import a Dynamo table that contains the StA parameters. This option will evaluate '
                           'the average and transform the Dynamo table to Scipion metadata format. and allows you to '
                           'perform post-StA processes (refinement and heterogeneity analysis).')
        group.addParam('tomBoxTable', params.PathParam, allowsNull=True,
                       condition='import_choice==%d' % IMPORT_TOMBOX_MTV,
                       label='Import a TOM-toolbox table (motive list) [Beta]',
                      help='import a TOM-toolbox table that contains the STA parameters. This option will evaluate '
                           'the average and transform the motive list to Scipion metadata format. and allows you to '
                           'perform post-StA processes (refinement and heterogeneity analysis).')
        group = form.addGroup('Subtomogram Averaging using Fast Rotational Matching (FRM)',
                              condition='StA_choice==%d'% PERFORM_STA)
        group.addParam('StartingReference', params.EnumParam,
                      choices=['start from scratch', 'browse for a volume file and use it as reference',
                               'select a volume from the workspace and use as reference'],
                      default=REFERENCE_NONE,
                      label='Starting reference', display=params.EnumParam.DISPLAY_COMBO,
                      help='Align from scratch of choose a template')
        group.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXISTS,
                      label="starting reference file",
                      help='Choose a starting reference from an external volume file')
        group.addParam('ReferenceImported', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_IMPORTED,
                      label="selected starting reference",
                      help='Choose an imported volume as a starting reference')
        group.addParam('applyMask', params.BooleanParam, label='Use a mask?', default=False,
                       help='A mask that can be applied on the reference without cropping it. The same mask will be'
                            ' applied on the aligned subtomograms at each iteration (do not apply this mask in advance)'
                       )
        group.addParam('Mask', params.PointerParam,
                       condition='applyMask',
                       pointerClass='Volume', allowsNull=True,
                       label="Select mask")
        group.addParam('NumOfIters', params.IntParam, default=10,
                      label='Number of iterations', help='How many times you want to iterate while performing'
                                                         ' subtomogram alignment and averaging.')
        group.addParam('WedgeMode', params.EnumParam,
                      choices=['Do not compensate', 'Compensate'],
                      default=WEDGE_MASK_THRE,
                      label='Missing-wedge Compensation', display=params.EnumParam.DISPLAY_COMBO,
                      help='Choose to compensate for the missing wedge if aligning subtomograms.'
                           ' However, if you are working with previously aligned subtomograms, then its better not to.')
        line = group.addLine('Low and high tilt values:', condition='WedgeMode==%d' % WEDGE_MASK_THRE,
                             help='The lower and upper tilt angles used in obtaining the tilt series')
        line.addParam('tiltLow', params.IntParam, default=-60,
                      label='Lower tilt value')
        line.addParam('tiltHigh', params.IntParam, default=60,
                      label='Upper tilt value')
        line = group.addLine('FRM parameters:',
                             help='The normalized frequency should be between 0 and 0.5 '
                                  'The more it is, the bigger the search frequency is, the more time it demands, '
                                  'keeping it as default is recommended. The maximum shift is a number between 1 and half the '
                                  'size of your volume. It represents the maximum distance searched in x,y and'
                                  ' z directions.'
                             )
        line.addParam('frm_freq', params.FloatParam, default=0.25,
                      label='Maximum searched frequency (0->0.5)')
        line.addParam('frm_maxshift', params.IntParam, default=10,
                      label='Maximum shift search (in pixels)',
                      help='')
        form.addParallelSection(threads=0, mpi=multiprocessing.cpu_count()//2-1)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        self.outputVolume = self._getExtraPath('final_average.mrc')
        self.outputMD = self._getExtraPath('final_md.xmd')

        self._insertFunctionStep('convertInputStep')
        if self.StA_choice.get() == PERFORM_STA:
            self._insertFunctionStep('doAlignmentStep')
        elif self.StA_choice.get() == COPY_STA and self.import_choice.get() == IMPORT_DYNAMO_TBL:
            self._insertFunctionStep('adaptDynamoStep', self.dynamoTable.get())
        elif self.StA_choice.get() == COPY_STA and self.import_choice.get() == IMPORT_TOMBOX_MTV:
            self._insertFunctionStep('adaptTomboxStep', self.tomBoxTable.get())
        else:
            self._insertFunctionStep('adaptXmippStep', self.xmippMD.get())
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
                    runProgram('xmipp_image_convert',args)
                else:
                    params = '-i %(imgPath)s --plus %(tempVol)s -o %(tempVol)s ' % locals()
                    runProgram('xmipp_image_operate', params)
            params = '-i %(tempVol)s --divide %(counter)s -o %(initialref)s ' % locals()
            runProgram('xmipp_image_operate', params)
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

            if self.applyMask.get():
                args += "--mask " + self.Mask.get().getFileName()

            self.runJob("xmipp_volumeset_align", args % locals(),
                        env = Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

            # By now, the alignment is done, the averaging should take place
            # However, if the alignemnt has missing wedge compensation, we shall update the metadata:
            if self.WedgeMode == WEDGE_MASK_THRE:
                mdImgs = md.MetaData(md_itr)
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
                mdImgs.write(md_itr)

            mdImgs = md.MetaData(md_itr)
            counter = 0

            for objId in mdImgs:
                counter = counter + 1

                imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
                rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
                tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
                psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

                x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
                y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
                z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

                tempVol = self._getExtraPath('temp.mrc')
                extra = self._getExtraPath()

                params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()
                runProgram('xmipp_transform_geometry', params)

                if counter == 1:
                    os.system("cp %(tempVol)s %(avr_itr)s" % locals())

                else:
                    params = '-i %(tempVol)s --plus %(avr_itr)s -o %(avr_itr)s ' % locals()
                    runProgram('xmipp_image_operate', params)

            params = '-i %(avr_itr)s --divide %(counter)s -o %(avr_itr)s ' % locals()
            runProgram('xmipp_image_operate', params)
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
        mdImgs.sort(md.MDL_ITEM_ID)
        mdImgs.write(self.outputMD)


    def adaptDynamoStep(self, dynamoTable):
        volumes_in = self.imgsFn
        volume_out = self.outputVolume
        md_out = self.outputMD
        from continuousflex.protocols.utilities.dynamo import tbl2metadata
        tbl2metadata(dynamoTable, volumes_in, md_out)

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

            tempVol = self._getExtraPath('temp.mrc')
            extra = self._getExtraPath()


            if first:
                print("Averaging based on Dynamo parameters")
                first = False

            params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                     ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            runProgram('xmipp_transform_geometry', params)

            if counter == 1:
                os.system("cp %(tempVol)s %(volume_out)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(volume_out)s -o %(volume_out)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(volume_out)s --divide %(counter)s -o %(volume_out)s ' % locals()
        runProgram('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())


    def adaptTomboxStep(self, Table):
        volumes_in = self.imgsFn
        volume_out = self.outputVolume
        md_out = self.outputMD
        from continuousflex.protocols.utilities.tombox import motivelist2metadata
        motivelist2metadata(Table, volumes_in, md_out)

        # Averaging based on the metadata:
        mdImgs = md.MetaData(md_out)
        counter = 0

        for objId in mdImgs:
            counter = counter + 1

            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

            tempVol = self._getExtraPath('temp.mrc')
            extra = self._getExtraPath()

            params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                     ' --shift %(x_shift)s %(y_shift)s %(z_shift)s' % locals()

            runProgram('xmipp_transform_geometry', params)

            if counter == 1:
                os.system("cp %(tempVol)s %(volume_out)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(volume_out)s -o %(volume_out)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(volume_out)s --divide %(counter)s -o %(volume_out)s ' % locals()
        runProgram('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())
         # Averaging is done
        pass


    def adaptXmippStep(self, Table):
        volumes_in = self.imgsFn
        volume_out = self.outputVolume
        md_out = Table

        # Averaging based on the metadata:
        mdImgs = md.MetaData(md_out)

        # if the volumes were aligned with angle_y=90 degrees, then rotate by 90 and inverse, then set angle y to 0
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

        mdImgs.write(self._getExtraPath('final_md.xmd'))
        counter = 0

        for objId in mdImgs:
            counter = counter + 1

            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

            tempVol = self._getExtraPath('temp.mrc')
            extra = self._getExtraPath()

            params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                 ' --shift %(x_shift)s %(y_shift)s %(z_shift)s' % locals()

            runProgram('xmipp_transform_geometry', params)

            if counter == 1:
                os.system("cp %(tempVol)s %(volume_out)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(volume_out)s -o %(volume_out)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(volume_out)s --divide %(counter)s -o %(volume_out)s ' % locals()
        runProgram('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())


    def createOutputStep(self):
        inputSet = self.inputVolumes.get()
        outvolume = Volume()
        outvolume.setSamplingRate(inputSet.getSamplingRate())
        outvolume.setFileName(self.outputVolume)
        self._defineOutputs(SubtomogramAverage=outvolume)


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex','CHEN2013235']

    def _methods(self):
        pass
