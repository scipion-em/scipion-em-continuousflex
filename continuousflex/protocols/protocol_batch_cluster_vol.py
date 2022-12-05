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
    """ Protocol executed when a cluster is created
    from NMA volumes and theirs deformations.
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
