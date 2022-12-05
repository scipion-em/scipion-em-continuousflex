# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# * Slavica Jonic (slavica.jonic@upmc.fr)
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
from pwem.objects import SetOfParticles, Volume, AtomStruct
from xmipp3.convert import writeSetOfParticles
from pwem.utils import runProgram
import pwem.emlib.metadata as md
import numpy as np


class FlexBatchProtNMACluster(BatchProtocol):
    """ Protocol executed when a cluster is created
    from NMA images and theirs deformations.
    """
    _label = 'nma cluster'
    
    def _defineParams(self, form):
        form.addHidden('inputNmaDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)
        
    #--------------------------- INSERT steps functions --------------------------------------------
        
    def _insertAllSteps(self):
        imagesMd = self._getExtraPath('images.xmd')
        outputVol = self._getExtraPath('reconstruction.vol')
        
        self._insertFunctionStep('convertInputStep', imagesMd)
        params = '-i %(imagesMd)s -o %(outputVol)s --fast' % locals()
        self._insertFunctionStep('reconstructStep', params)
        self._insertFunctionStep('centroidPdbStep')
        self._insertFunctionStep('createOutputStep', outputVol)
        
    #--------------------------- STEPS functions --------------------------------------------   
        
    def convertInputStep(self, imagesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        inputSet = self.inputNmaDimred.get().getInputParticles()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(inputSet)
        
        tmpSet = SetOfParticles(filename=self.sqliteFile.get())        
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()
        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)
        
        writeSetOfParticles(partSet, imagesMd)

        # Add the NMA displacement to clusters XMD files
        md_file_nma = md.MetaData(self.inputNmaDimred.get().getParticlesMD())
        md_file_org = md.MetaData(imagesMd)
        for objID in md_file_org:
            # if image name is the same, we add the nma displacement from nma to org
            id_org = md_file_org.getValue(md.MDL_ITEM_ID, objID)
            for j in md_file_nma:
                id_nma = md_file_nma.getValue(md.MDL_ITEM_ID, j)
                print(id_nma)
                if id_org == id_nma:
                    displacements = md_file_nma.getValue(md.MDL_NMA, j)
                    md_file_org.setValue(md.MDL_NMA, displacements, objID)
                    break
        md_file_org.write(imagesMd)


    def reconstructStep(self, params):
        runProgram('xmipp_reconstruct_fourier_accel', params)


    def centroidPdbStep(self):
        imagesMd = self._getExtraPath('images.xmd')
        md_file = md.MetaData(imagesMd)
        deformations = []
        for j in md_file:
            defor = md_file.getValue(md.MDL_NMA, j)
            if defor:
                deformations.append(defor)
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
        vol.setSamplingRate(self.outputParticles.getSamplingRate())
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
        return self.inputNmaDimred.get().getInputModes()

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
    
