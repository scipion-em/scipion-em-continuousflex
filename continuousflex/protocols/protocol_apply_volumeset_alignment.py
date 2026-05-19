# **************************************************************************
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

from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath, copyFile
from os.path import basename
from pwem.utils import runProgram
from os.path import exists, basename, abspath, relpath, join, splitext
import pwem.objects as emobj
from xmipp3.convert import writeSetOfVolumes, readSetOfVolumes
from .convert import eulerAngles2matrix, matrix2eulerAngles
import numpy as np
from pyworkflow.utils.path import makePath, copyFile
import json
from ast import literal_eval as make_tuple
import os
from pwem.constants import ALIGN_3D

class FlexProtApplyVolSetAlignment(ProtAnalysis3D):
    """
    Applies subtomogram alignment parameters obtained from external
    subtomogram averaging workflows to a set of 3D volumes. The protocol
    standardizes the spatial orientation of subtomograms so they can be
    consistently interpreted, compared, visualized, or used in downstream
    structural analysis.

    AI Generated:

    Apply Subtomogram Alignment (FlexProtApplyVolSetAlignment) - User Manual
        Overview

        The Apply Subtomogram Alignment protocol transfers alignment
        parameters generated during subtomogram averaging workflows onto
        individual subtomograms or reconstructed volumes. Its main purpose
        is to place all input volumes into a common spatial reference frame
        so that structural variability, conformational organization, or
        biological patterns can be studied consistently across a dataset.

        In cryo-electron tomography workflows, subtomograms are often
        aligned using external software packages specialized in subtomogram
        averaging and refinement. However, the resulting transformations are
        not always directly applied to the original volumes within the same
        environment. This protocol bridges that gap by importing alignment
        information from several commonly used tomography platforms and
        applying those transformations to the input data.

        Biological Context

        For biological users, subtomogram alignment is essential when
        studying macromolecular complexes in their native cellular
        environment. Proper alignment enables direct comparison of
        structures extracted from different cellular regions, experimental
        conditions, or conformational states. Without a consistent spatial
        orientation, biological interpretation becomes difficult because
        structural variability may reflect geometric inconsistency rather
        than genuine molecular differences.

        This protocol is particularly useful after subtomogram averaging
        refinement, where the averaging software has already estimated the
        orientation and position of each subtomogram relative to a
        consensus structure. Applying these alignments back onto the
        original subtomograms allows users to visualize aligned particles,
        perform focused analyses, generate curated datasets, or prepare
        inputs for classification and heterogeneity studies.

        Supported Alignment Sources

        The protocol supports importing alignment information from several
        major tomography processing ecosystems. This flexibility is valuable
        in collaborative environments where datasets may originate from
        different facilities or processing pipelines.

        Xmipp alignment metadata can be imported directly from metadata
        files containing rotational and translational parameters. This mode
        is particularly convenient for users already working within Scipion
        and Xmipp-based tomography workflows.

        EMAN alignment information can also be imported from refinement
        metadata generated during subtomogram alignment procedures. This
        allows users to continue processing EMAN-refined datasets within a
        unified analysis environment.

        Dynamo tables are additionally supported, enabling integration with
        Dynamo-based subtomogram averaging projects. This is especially
        important for users studying large in situ assemblies where Dynamo
        remains a widely adopted refinement platform.

        The protocol is designed with extensibility in mind so that
        additional tomography alignment formats may be incorporated into
        future workflows.

        Input Requirements and Consistency

        The protocol requires a set of input subtomograms together with
        alignment parameters describing their orientations and positional
        shifts. The number of alignment records must match the number of
        input volumes to ensure that every subtomogram receives the correct
        transformation.

        From a biological perspective, maintaining consistency between the
        alignment metadata and the subtomogram dataset is critical. Applying
        transformations to mismatched particles can generate misleading
        structural interpretations and compromise downstream analyses.

        Ideally, the subtomograms should already share a consistent voxel
        size and box dimensions before alignment application. Significant
        differences in sampling or volume dimensions may complicate
        comparison and visualization after transformation.

        Spatial Transformations and Coordinate Systems

        The protocol applies rigid-body spatial transformations that include
        rotations and translational shifts. These operations reposition each
        subtomogram into a standardized orientation relative to the
        alignment reference used during subtomogram averaging.

        In tomography workflows, coordinate system conventions may differ
        between software packages. The protocol therefore handles the
        interpretation of orientation conventions internally to ensure that
        imported transformations remain biologically meaningful when applied
        within the current environment.

        Particular attention is given to subtomogram orientations affected
        by missing wedge geometry, since this artifact can strongly
        influence alignment interpretation in cryo-electron tomography.
        Correct handling of orientation conventions improves consistency
        between visualization and refinement environments.

        Outputs and Biological Interpretation

        The primary output is a set of aligned subtomograms expressed in a
        common coordinate frame. Once aligned, volumes become easier to
        compare visually and quantitatively because corresponding structural
        regions occupy consistent spatial positions.

        Biologically, aligned subtomograms can reveal conserved structural
        features across particles while making conformational differences
        easier to interpret. This is particularly important in studies of
        flexible molecular assemblies, membrane-associated complexes, or
        large cellular machineries observed directly inside cells.

        The aligned outputs can also serve as inputs for downstream
        classification, averaging, dimensionality reduction, variability
        analysis, or visualization workflows. Because all subtomograms share
        a unified orientation, subsequent analyses become more robust and
        easier to interpret biologically.

        Practical Recommendations

        In routine tomography practice, users should first verify that the
        imported alignment parameters correspond exactly to the subtomogram
        dataset being processed. Small inconsistencies in ordering or file
        correspondence can propagate into major structural interpretation
        errors.

        It is also advisable to visually inspect a subset of aligned
        subtomograms after processing. Successful alignment should place
        major structural landmarks into similar orientations across the
        dataset. Unexpected variability may indicate problems in the
        original averaging refinement, coordinate conventions, or metadata
        consistency.

        When combining subtomograms refined in different software packages,
        users should remain aware that alignment conventions may differ
        slightly between platforms. Careful validation and visualization are
        therefore recommended before proceeding to biological conclusions.

        Final Perspective

        For cryo-electron tomography studies, applying subtomogram
        alignments is a crucial step that transforms independently oriented
        cellular particles into a coherent structural dataset. By placing
        subtomograms into a shared spatial framework, the protocol enables
        more reliable structural interpretation, clearer visualization of
        biological variability, and improved integration between tomography
        processing environments.
    """
    _label = 'apply subtomogram alignment'
    IMPORT_FROM_XMIPP=0
    IMPORT_FROM_EMAN=1
    IMPORT_FROM_DYNAMO=2
    IMPORT_FROM_TOMBOX=3

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      label="Input volume(s)", important=True,
                      help='Select volumes')

        form.addParam('importFrom', params.EnumParam, default=self.IMPORT_FROM_XMIPP,
                      allowsNull=True,
                      choices=['XMIPP', 'EMAN', 'DYNAMO', 'TOMBOX'],
                      label='import STA alignment from',
                      help='Select the alignment files to apply to the volumes')

        form.addParam('xmdFile', params.FileParam,
                      condition='(importFrom == %d)' % self.IMPORT_FROM_XMIPP,
                      label='Input Xmipp Metatada file',
                      help="Select the XMD file containing subtomograms and alignment ")

        form.addParam('inputVolsDynamo', params.PointerParam,
                      condition='(importFrom == %d)' % self.IMPORT_FROM_DYNAMO, pointerClass='SetOfVolumes',
                      label='Input volumes',
                      help="Select a set of volumes")
        form.addParam('dynamoTable', params.PathParam,
                      condition='(importFrom == %d)' % self.IMPORT_FROM_DYNAMO,
                      label='Dynamo Table [Beta]',
                      help="import a Dynamo table that contains the StA parameters. ")

        form.addParam('emanJSON', params.PathParam, allowsNull=True,
                       condition='importFrom==%d' % self.IMPORT_FROM_EMAN,
                       label='Import a JSON file from EMAN [Beta]',
                       help='import a JSON file that contains the STA parameters. ')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('applyAlignment')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        if  self.importFrom == self.IMPORT_FROM_XMIPP:
            volSet = self.inputFromXmipp()
        elif  self.importFrom == self.IMPORT_FROM_EMAN:
            volSet = self.inputFromEman()
        elif  self.importFrom == self.IMPORT_FROM_DYNAMO:
            volSet = self.inputFromDynamo()
        elif  self.importFrom == self.IMPORT_FROM_TOMBOX:
            volSet = self.inputFromTombox()
        else:
            raise NotImplementedError("")

        inputVols = self.inputVolumes.get()

        if inputVols.getSize() == volSet.getSize():
            # Write a metadata with the volumes
            iter1 = volSet.iterItems()
            iter2 = inputVols.iterItems()
            inputset = self._createSetOfVolumes("inputSet")
            inputset.setSamplingRate(inputVols.getSamplingRate())
            inputset.setAlignment(ALIGN_3D)

            for i in range(volSet.getSize()):
                p1 = iter1.__next__()
                p2 = iter2.__next__()
                p2.setTransform(p1.getTransform())
                inputset.append(p2)
            xmipp3.convert.writeSetOfVolumes(inputset, self._getExtraPath('volumes.xmd'))
        else:
            raise RuntimeError("The number of volumes and STA parameters mismatch")




    def inputFromXmipp(self):

        mdImgs = md.MetaData(self.xmdFile)
        flag = None
        try:
            flag = mdImgs.getValue(md.MDL_ANGLE_Y, 1)
        except:
            pass

        if flag == 90:
            mdImgs = md.MetaData(self.xmdFile)
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
        return self.createVolSetSubtomo(mdImgs)

    def inputFromEman(self):
        Table = self.emanJSON.get()

        with open(Table, "r") as f:
            jf = json.load(f)
        n_data = len(jf)

        index = []
        fname = []
        matrices = []
        for i in jf:
            fname_i, index_i = make_tuple(i)
            index.append(index_i)
            fname.append(fname_i)
            mat = np.array(json.loads(jf[i]["xform.align3d"]["matrix"]), dtype=np.float64).reshape(3, 4)
            matrices.append(matrix2eulerAngles(mat))
        matrices = np.array(matrices)

        for i in range(n_data):
            fileext = os.path.splitext(fname[i])[1]
            if fileext == ".lst":
                with open(fname[i], "r") as f:
                    for line in f:
                        if not line.startswith('#'):
                            spl = line.split()
                            if int(spl[0]) == index[i]:
                                fname[i] = spl[1]
                                break
            elif fileext == ".hdf" or fileext == ".mrc" or fileext == ".mrcs" or fileext == ".vol" or fileext == ".spi":
                pass
            else:
                raise RuntimeError("Unkown file type for subtomograms")

        volSet = self._createSetOfVolumes()
        volSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())

        for i in range(n_data):

            imgPath = "%s@%s" % (str(index[i] + 1).zfill(6), abspath(fname[i]))
            transform = emobj.Transform()
            transform.setMatrix(matrices[i])
            vol = emobj.Volume()
            vol.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            vol.cleanObjId()
            vol.setTransform(transform)
            vol.setLocation(imgPath)
            volSet.append(vol)
        volSet.setAlignment3D()
        return volSet

    def inputFromDynamo(self):
        from continuousflex.protocols.utilities.dynamo import tbl2metadata

        volumes_in = self._getExtraPath('input.xmd')
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), volumes_in)
        tbl2metadata(self.dynamoTable.get(), volumes_in, md_out)

        mdImgs = md.MetaData(md_out)
        return self.createVolSetSubtomo(mdImgs)

    def inputFromTombox(self):
        raise NotImplementedError()

    def createVolSetSubtomo(self, mdImgs):
        volSet = self._createSetOfVolumes()
        volSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())

        for objId in mdImgs:

            # imgPath = abspath(mdImgs.getValue(md.MDL_IMAGE, objId))
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)

            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)
            matrix = eulerAngles2matrix(rot, tilt, psi, x_shift, y_shift, z_shift)

            transform = emobj.Transform()
            transform.setMatrix(matrix)

            vol = emobj.Volume()
            vol.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            vol.cleanObjId()
            vol.setTransform(transform)
            # vol.setLocation(imgPath)
            volSet.append(vol)
        volSet.setAlignment3D()
        return volSet


    def applyAlignment(self):
        makePath(self._getExtraPath() + '/aligned')
        tempdir = self._getTmpPath()
        mdImgs = md.MetaData(self._getExtraPath('volumes.xmd'))
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath()+'/aligned/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # rotate 90 around y, align, then rotate -90 to get to neutral
            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            runProgram('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            params += ' --inverse '
            runProgram('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned.xmd')
        mdImgs.write(self.fnaligned)


    def createOutputStep(self):
        partSet = self._createSetOfVolumes('aligned')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes_aligned.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(outputVolumes=partSet)


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex']

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
