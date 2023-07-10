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


class FlexProtApplyVolSetAlignment(ProtAnalysis3D):
    """ Protocol for subtomogram alignment after STA """
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
            self.inputFromXmipp()
        elif  self.importFrom == self.IMPORT_FROM_EMAN:
            self.inputFromEman()
        elif  self.importFrom == self.IMPORT_FROM_DYNAMO:
            self.inputFromDynamo()
        elif  self.importFrom == self.IMPORT_FROM_TOMBOX:
            self.inputFromTombox()
        else:
            raise NotImplementedError("")

        if self.inputVolumes.get().getSize() == self.volSet.getSize():
            # Write a metadata with the volumes
            xmipp3.convert.writeSetOfVolumes(self.volSet, self._getExtraPath('volumes.xmd'))
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

        mdImgs.write(self._getExtraPath('output.xmd'))
        self.createVolSetSubtomo(mdImgs)

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
        self.volSet = volSet

    def inputFromDynamo(self):
        from continuousflex.protocols.utilities.dynamo import tbl2metadata

        volumes_in = self._getExtraPath('input.xmd')
        xmipp3.convert.writeSetOfVolumes(self.inputVolsDynamo.get(), volumes_in)
        md_out =self._getExtraPath('output.xmd')
        tbl2metadata(self.dynamoTable.get(), volumes_in, md_out)

        mdImgs = md.MetaData(md_out)
        self.createVolSetSubtomo(mdImgs)

    def inputFromTombox(self):
        raise NotImplementedError()

    def createVolSetSubtomo(self, mdImgs):
        volSet = self._createSetOfVolumes()
        volSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())

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
            vol.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            vol.cleanObjId()
            vol.setTransform(transform)
            vol.setLocation(imgPath)
            volSet.append(vol)
        volSet.setAlignment3D()
        self.volSet=volset


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
