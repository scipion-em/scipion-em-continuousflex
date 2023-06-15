# **************************************************************************
# * Authors:   Rémi Vuillemot remi.vuillemot@upmc.fr
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

from pwem.protocols import ProtImportFiles
import xmipp3.convert
import pyworkflow
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
import pwem.objects as emobj
from pwem import emlib
from os.path import exists, basename, abspath, relpath, join, splitext
from xmipp3.convert import writeSetOfVolumes, readSetOfVolumes
from .convert import eulerAngles2matrix, matrix2eulerAngles
import numpy as np
from pyworkflow.utils.path import makePath, copyFile
from os.path import basename
from pwem.utils import runProgram
import json
from ast import literal_eval as make_tuple
import os

class FlexProtImportSubtomogram(ProtImportFiles):
    """ Protocol for importing subtomograms"""
    _label = 'import subtomogram'
    IMPORT_FROM_XMIPP=1
    IMPORT_FROM_EMAN=2
    IMPORT_FROM_DYNAMO=3
    IMPORT_FROM_TOMBOX=4

    def _defineImportParams(self, form):
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

        form.addParam('samplingRate', params.FloatParam, label='Voxel size (sampling rate) Å/px')

    def _insertAllSteps(self):
        if self.importFrom == self.IMPORT_FROM_FILES:
            self._insertFunctionStep(self.importFromFileStep,
                                     self.getPattern(),
                                     self.samplingRate.get())
        elif  self.importFrom == self.IMPORT_FROM_XMIPP:
            self._insertFunctionStep(self.inputFromXmipp)

        elif  self.importFrom == self.IMPORT_FROM_EMAN:
            self._insertFunctionStep(self.inputFromEman)
        elif  self.importFrom == self.IMPORT_FROM_DYNAMO:
            self._insertFunctionStep(self.inputFromDynamo)
        elif  self.importFrom == self.IMPORT_FROM_TOMBOX:
            self._insertFunctionStep(self.inputFromTombox)
        else:
            raise NotImplementedError("")


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

        mdImgs.write(self._getExtraPath('output.xmd'))
        self.createOutputSubtomo(mdImgs)

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
        volSet.setSamplingRate(self.samplingRate.get())

        for i in range(n_data):

            imgPath = "%s@%s" % (str(index[i] + 1).zfill(6), abspath(fname[i]))
            transform = emobj.Transform()
            transform.setMatrix(matrices[i])
            vol = emobj.Volume()
            vol.setSamplingRate(self.samplingRate.get())
            vol.cleanObjId()
            vol.setTransform(transform)
            vol.setLocation(imgPath)
            volSet.append(vol)
        volSet.setAlignment3D()
        self._defineOutputs(**{"ImportSubtomo": volSet})

    def inputFromDynamo(self):
        from continuousflex.protocols.utilities.dynamo import tbl2metadata

        volumes_in = self._getExtraPath('input.xmd')
        xmipp3.convert.writeSetOfVolumes(self.inputVolsDynamo.get(), volumes_in)
        md_out =self._getExtraPath('output.xmd')
        tbl2metadata(self.dynamoTable.get(), volumes_in, md_out)

        mdImgs = md.MetaData(md_out)
        self.createOutputSubtomo(mdImgs)

    def inputFromTombox(self):
        raise NotImplementedError()

    def importFromFileStep(self, pattern, samplingRate):
        """ Copy images matching the filename pattern
        Register other parameters.
        """
        volSet = self._createSetOfVolumes()
        vol = emobj.Volume()

        self.info("Using pattern: '%s'" % pattern)

        # Create a Volume template object
        vol.setSamplingRate(samplingRate)

        imgh = emlib.image.ImageHandler()

        volSet.setSamplingRate(samplingRate)

        for fileName, fileId in self.iterFiles():
            x, y, z, n = imgh.getDimensions(fileName)
            if fileName.endswith('.mrc') or fileName.endswith('.map'):
                fileName += ':mrc'
                if z == 1 and n != 1:
                    zDim = n
                    n = 1
                else:
                    zDim = z
            else:
                zDim = z
            origin = emobj.Transform()
            origin.setShifts(x / -2. * samplingRate,
                                 y / -2. * samplingRate,
                                 zDim / -2. * samplingRate)

            vol.setOrigin(origin)  # read origin from form

            newFileName = abspath(self._getVolumeFileName(fileName))

            if fileName.endswith(':mrc'):
                fileName = fileName[:-4]

            pwutils.createAbsLink(fileName, newFileName)
            newFileName = relpath(newFileName)
            for index in range(1, n + 1):
                vol.cleanObjId()
                vol.setLocation(index, newFileName)
                volSet.append(vol)

        self._defineOutputs(**{"ImportSubtomo": volSet})

    def createOutputSubtomo(self, mdImgs):
        volSet = self._createSetOfVolumes()
        volSet.setSamplingRate(self.samplingRate.get())

        for objId in mdImgs:

            imgPath = abspath(mdImgs.getValue(md.MDL_IMAGE, objId))
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
            vol.setSamplingRate(self.samplingRate.get())
            vol.cleanObjId()
            vol.setTransform(transform)
            vol.setLocation(imgPath)
            volSet.append(vol)
        volSet.setAlignment3D()
        self._defineOutputs(**{"ImportSubtomo": volSet})

    def _getVolumeFileName(self, fileName, extension=None):
        if extension is not None:
            baseFileName = "import_" + basename(fileName).split(".")[0] + ".%s" % extension
        else:
            baseFileName = "import_" + basename(fileName).split(":")[0]

        return self._getExtraPath(baseFileName)
    def _getImportChoices(self):
        """ Return a list of possible choices
        from which the import can be done.
        (usually packages formats such as: xmipp3, eman2, relion...etc.)
        """
        return ['files',"xmipp", "eman", "dynamo","tomobox"]