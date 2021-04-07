# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# *           Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# * IMPMC, UPMC Sorbonne University
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
# **************************************************************************

from os.path import basename

from pwem.convert.atom_struct import cifToPdb
from pyworkflow.utils import replaceBaseExt, replaceExt

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
from pwem.objects import AtomStruct, Volume
import xmipp3
import os
import numpy as np


np.random.seed(0)

NMA_ALIGNMENT_WAV = 0
NMA_ALIGNMENT_PROJ = 1

MODE_RELATION_LINEAR = 0
MODE_RELATION_3CLUSTERS = 1
MODE_RELATION_MESH = 2
MODE_RELATION_RANDOM = 3

MISSINGWEDGE_YES = 0
MISSINGWEDGE_NO = 1

ROTATION_SHIFT_YES = 0
ROTATION_SHIFT_NO = 1

RECONSTRUCTION_FOURIER = 0
RECONSTRUCTION_WBP = 1

NOISE_CTF_YES = 0
NOISE_CTF_NO = 1

FULL_TOMOGRAM_YES= 0
FULL_TOMOGRAM_NO = 1

class FlexProtSynthesizeSubtomo(ProtAnalysis3D):
    """ Protocol for synthesizing subtomograms. """
    _label = 'synthesize subtomograms'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', NumericRangeParam,
                      label="Modes selection",
                      default='7-8',
                      help='Select the normal modes that will be used for image analysis. \n'
                           'It is usually two modes that should be selected, unless if the relationship is linear or random.\n'
                           'You have several ways to specify the modes.\n'
                           ' Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('modeRelationChoice', params.EnumParam, default=MODE_RELATION_LINEAR,
                      choices=['Linear relationship', '3 Clusters', 'Mesh', 'Random'],
                      label='Relationship between the modes',
                      help='linear relationship: all the selected modes will have equal amplitudes. \n'
                           '3 clusters: the volumes will be devided exactly into three classes.\n'
                           'Mesh: the amplitudes will be in a mesh shape (mesh size is square of what mesh step).\n'
                           'Random: all the amplitudes will be random in the given range')
        form.addParam('centerPoint', params.IntParam, default=100,
                      condition='modeRelationChoice==%d' % MODE_RELATION_3CLUSTERS,
                      label='Center point',
                      help='This number will be used to determine the distance between the clusters'
                           'center1 = (-center_point, 0)'
                           'center2 = (center_point, 0)'
                           'center3 = (0, center_point)')
        form.addParam('modesAmplitudeRange', params.IntParam, default=150,
                      condition='modeRelationChoice==%d or modeRelationChoice==%d or modeRelationChoice==%d' % (MODE_RELATION_LINEAR,MODE_RELATION_MESH, MODE_RELATION_RANDOM),
                      label='Amplitude range N --> [-N, N]',
                      help='Choose the number N for which the generated normal mode amplitudes are in the range of [-N, N]')
        form.addParam('meshRowPoints', params.IntParam, default=6,
                      condition='modeRelationChoice==%d' % MODE_RELATION_MESH,
                      label='Mesh number of steps',
                      help='This number will be the number of points in the row and the column (mesh shape will be size*size)')
        form.addParam('numberOfVolumes', params.IntParam, default=36,
                      label='Number of volumes',
                      condition='modeRelationChoice!=%d'% MODE_RELATION_MESH,
                      help='Number of volumes that will be generated')
        form.addParam('samplingRate', params.FloatParam, default=2.2,
                      label='Sampling Rate',
                      help='The sampling rate (voxel size in Angstroms)')
        form.addParam('volumeSize', params.IntParam, default=64,
                      label='Volume Size',
                      help='Volume size in voxels (all volumes will be cubes)')

        form.addSection(label='Missing wedge parameters')
        form.addParam('missingWedgeChoice', params.EnumParam, default=MISSINGWEDGE_YES,
                      choices=['Simulate missing wedge artefacts', 'No missing wedge'],
                      label='Missing Wedge choice',
                      help='We can either simulate missing wedge artefacts by generating with tilt range or the tilt range will be full -90 to 90')
        form.addParam('tiltStep', params.IntParam, default=1,
                      label='tilt step angle',
                      help='tilting step in the tilt range (examples 1 , 2, 4 degree)')
        form.addParam('tiltLow', params.IntParam, default=-60,
                      condition='missingWedgeChoice==%d' % MISSINGWEDGE_YES,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        form.addParam('tiltHigh', params.IntParam, default=60,
                      condition='missingWedgeChoice==%d' % MISSINGWEDGE_YES,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')

        form.addSection(label='Noise and CTF')
        form.addParam('noiseCTFChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Apply Noise and CTF',
                      help='If not selected, the CTF will not be simulated')
        form.addParam('targetSNR', params.FloatParam, default=0.1,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='Tagret SNR',
                      help='The signal to noise ratio for the noise')
        form.addParam('ctfVoltage', params.FloatParam, default=200.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Voltage',
                      help='Simualted voltage in the microscope in kV')
        form.addParam('ctfSphericalAberration', params.FloatParam, default=2,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Spherical Aberration',
                      help='Microscope attribute')
        form.addParam('ctfMagnification', params.FloatParam, default=50000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Magnification',
                      help='Maginification in the microscope')
        form.addParam('ctfDefocusU', params.FloatParam, default=-5000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF DefocusU',
                      help='defocus value (keep the value for DefocusV for simplicity)')
        form.addParam('ctfDefocusV', params.FloatParam, default=-5000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF DefocusV',
                      help='defocus value (keep the value for DefocusU for simplicity)')
        form.addParam('ctfQ0', params.FloatParam, default=-0.112762,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Q0',
                      help='Microscope attribute')

        form.addSection('Reconstruction')
        form.addParam('reconstructionChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Fourier interpolation', ' Weighted BackProjection'],
                      label='Reconstruction method',
                      help='Tomographic reconstruction method')

        form.addSection('Rigid body variability')
        form.addParam('rotationShiftChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Simulate Rotations and Shifts',
                      help='If yes, the volumes will be rotated and shifted randomly in the range,'
                           ' otherwise no rotations nor shifts will be applied')
        form.addParam('maxShift', params.FloatParam, default=5.0,
                      label='Maximum Shift',
                      help='The maximum shift from the center that will be applied')

        form.addSection('Generate full tomogram')
        form.addParam('fullTomogramChoice', params.EnumParam, default=FULL_TOMOGRAM_NO,
                      choices=['Yes', 'No'],
                      label='Generate full tomogram',
                      help='If yes, the generated volumes will be put in a big volume, a tomogram will be generated')
        form.addParam('numberOfTomograms', params.IntParam, default=1,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Number of tomograms',
                      help='We can distribute the volumes on several tomograms')
        form.addParam('tomoSizeX', params.IntParam, default=512,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size X',
                      help='X dimension of the tomogram')
        form.addParam('tomoSizeY', params.IntParam, default=512,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size Y',
                      help='Y dimension of the tomogram')
        form.addParam('tomoSizeZ', params.IntParam, default=128,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size Z',
                      help='Z dimension of the tomogram')
        form.addParam('boxSize', params.IntParam, default=64,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Box Size',
                      help='The distance where volumes inside the tomogram will not overlap,'
                           ' the bigger the more seperated the molecules inside')

        # form.addParallelSection(threads=0, mpi=8)

        # --------------------------- INSERT steps functions --------------------------------------------

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """
        return self.inputModes.get().getPdb()

    def _insertAllSteps(self):
        self._insertFunctionStep("generate_deformations")
        self._insertFunctionStep("generate_volume_from_pdb")

        if self.rotationShiftChoice == ROTATION_SHIFT_YES:
            self._insertFunctionStep("generate_rotation_and_shift")

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            self._insertFunctionStep("create_phantom")
            self._insertFunctionStep("map_volumes_to_tomogram")

        self._insertFunctionStep("project_volumes")

        if self.noiseCTFChoice == NOISE_CTF_YES:
            self._insertFunctionStep("apply_noise_and_ctf")

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            pass
        else:
            self._insertFunctionStep("reconstruct")

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def generate_deformations(self):
        # use the input relationship between the modes to generate normal mode amplitudes metadata
        fnPDB = self.inputModes.get().getPdb().getFileName()
        fnModeList = replaceExt(self.inputModes.get().getFileName(),'xmd')
        modeAmplitude = self.modesAmplitudeRange.get()
        meshRowPoints = self.meshRowPoints.get()
        numberOfModes = self.inputModes.get().getSize()
        modeSelection = np.array(getListFromRangeString(self.modeList.get()))
        deformationFile = self._getExtraPath('GroundTruth.xmd')
        subtomogramMD = md.MetaData()
        # these vairables for alternating the generation when 3 clusters are selected
        cluster1 = cluster2 = cluster3 = False
        # these variables for the generaton of a mesh if selected
        XX, YY = np.meshgrid(np.linspace(start=-modeAmplitude, stop=modeAmplitude, num=meshRowPoints),
                             np.linspace(start=modeAmplitude, stop=-modeAmplitude, num=meshRowPoints))
        mode7_samples = XX.reshape(-1)
        mode8_samples = YY.reshape(-1)

        # iterate over the number of outputs (if mesh this has to be calculated)
        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()
        for i in range(numberOfVolumes):
            deformations = np.zeros(numberOfModes)

            if self.modeRelationChoice == MODE_RELATION_LINEAR:
                amplitude = self.modesAmplitudeRange.get()
                deformations[modeSelection - 1] = np.ones(len(modeSelection))*np.random.uniform(-amplitude, amplitude)
            elif self.modeRelationChoice == MODE_RELATION_3CLUSTERS:
                center_point = self.centerPoint.get()
                center1 = (-center_point, 0)
                center2 = (center_point, 0)
                center3 = (0, center_point)
                if not(cluster1 or cluster2 or cluster3):
                    cluster1 = True
                if cluster3:
                    deformations[modeSelection - 1] = center3
                    cluster3 = False
                if cluster2:
                    deformations[modeSelection - 1] = center2
                    cluster2 = False
                    cluster3 = True
                if cluster1:
                    deformations[modeSelection - 1] = center1
                    cluster1 = False
                    cluster2 = True
            elif self.modeRelationChoice == MODE_RELATION_RANDOM:
                amplitude=self.modesAmplitudeRange.get()
                deformations[modeSelection-1] = np.random.uniform(-amplitude, amplitude, len(modeSelection))
            elif self.modeRelationChoice == MODE_RELATION_MESH:
                new_point=(mode7_samples[i],mode8_samples[i])
                deformations[modeSelection - 1] = new_point

            # we won't keep the first 6 modes
            deformations = deformations[6:]

            params = " --pdb " + fnPDB
            params+= " --nma " + fnModeList
            params+= " -o " + self._getExtraPath(str(i+1).zfill(5)+'_deformed.pdb')
            params+= " --deformations " + ' '.join(str(i) for i in deformations)
            self.runJob('xmipp_pdb_nma_deform', params)

            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1).zfill(5)+'_reconstructed'+'.vol'), subtomogramMD.addObject())
            subtomogramMD.setValue(md.MDL_NMA, list(deformations), i+1)

        subtomogramMD.write(deformationFile)


    def generate_volume_from_pdb(self):
        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()
        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.pdb')
            params += " --sampling " + str(self.samplingRate.get())
            params += " --size " + str(self.volumeSize.get())
            params += " -v 0 --centerPDB "
            self.runJob('xmipp_volume_from_pdb', params)

    def generate_rotation_and_shift(self):
        subtomogramMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()

        for i in range(numberOfVolumes):
            rot1 = 360 * np.random.uniform()
            tilt1 = 180 * np.random.uniform()
            psi1 = 360 * np.random.uniform()
            shift_x1 = self.maxShift.get() * np.random.uniform(-1,1)
            shift_y1 = self.maxShift.get() * np.random.uniform(-1,1)
            shift_z1 = self.maxShift.get() * np.random.uniform(-1,1)

            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.vol')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.vol')
            params += " --rotate_volume euler " + str(rot1) + ' ' + str(tilt1) + ' ' + str(psi1)
            params += " --shift " + str(shift_x1) + ' ' + str(shift_y1) + ' ' + str(shift_z1)
            params += " --dont_wrap "
            self.runJob('xmipp_transform_geometry', params)

            subtomogramMD.setValue(md.MDL_SHIFT_X, shift_x1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Y, shift_y1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Z, shift_z1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_ROT, rot1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_TILT, tilt1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_PSI, psi1, i + 1)
        subtomogramMD.write(self._getExtraPath('GroundTruth.xmd'))

    def create_phantom(self):
        tomoSizeX = self.tomoSizeX.get()
        tomoSizeY = self.tomoSizeY.get()
        tomoSizeZ = self.tomoSizeZ.get()
        with open(self._getExtraPath('tomogram.param'), 'a') as file:
            file.write(
                "\n".join([
                        "# XMIPP_STAR_1 *",
                        "data_block1",
                        " _dimensions3D  '%(tomoSizeX)s %(tomoSizeY)s %(tomoSizeZ)s'" % locals(),
                        " _phantomBGDensity  0",
                        " _scale  1",
                        "data_block2",
                        "loop_",
                        " _featureType",
                        " _featureOperation",
                        " _featureDensity",
                        " _featureCenter",
                        " _featureSpecificVector",
                        " cub + 0.0 '0 0 0' '0 0 0 0 0 0'"]))

        # generate a phantom for each tomograms
        for i in range(self.numberOfTomograms.get()):
            params = " -i " + self._getExtraPath('tomogram.param')
            params += " -o " + self._getExtraPath(str(i+1).zfill(5) +'_tomogram.vol')
            self.runJob('xmipp_phantom_create', params)

    def map_volumes_to_tomogram(self):
        tomoSizeX = self.tomoSizeX.get()
        tomoSizeY = self.tomoSizeY.get()
        tomoSizeZ = self.tomoSizeZ.get()
        boxSize = self.boxSize.get()

        # create a 2D grid of the tomogram at the size of a box
        boxGrid = np.mgrid[boxSize // 2: tomoSizeX - boxSize // 2 + 1 :  boxSize,
                           boxSize // 2: tomoSizeY - boxSize // 2 + 1:  boxSize]

        #get the grid positions as an array
        numberOfBoxes = boxGrid.shape[1]*boxGrid.shape[2]
        boxPositions = boxGrid.reshape(2, numberOfBoxes).T

        # The number of particles per tomogram is the integer division of the number of volumes
        #  and the number of tomograms
        numberOfTomograms = self.numberOfTomograms.get()

        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()

        particlesPerTomogram = numberOfVolumes//numberOfTomograms

        for t in range(numberOfTomograms):
            # Shuffle the positions order to fill the tomogram boxes in random order
            np.random.shuffle(boxPositions)

            # if the number of particles per tomograms is > of the number of boxes, some particles are ignored
            for i in range(np.min([particlesPerTomogram, numberOfBoxes])):

                # Create a metadata file per volumes mapped per tomograms
                tomogramMapMD = md.MetaData()
                tomogramMapMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1).zfill(5) + '.vol'), tomogramMapMD.addObject())
                tomogramMapMD.setValue(md.MDL_XCOOR, int(boxPositions[i,0]), 1)
                tomogramMapMD.setValue(md.MDL_YCOOR, int(boxPositions[i,1]), 1)
                tomogramMapMD.setValue(md.MDL_ZCOOR, int(np.random.randint(boxSize // 2, tomoSizeZ - boxSize // 2)), 1)
                tomogramMapMD.write( self._getExtraPath(str(t+1).zfill(5) +"_"+str(i+1).zfill(5) +'_tomogram_map.xmd'))

                # Map each volumes to the right tomogram
                params = " -i " + self._getExtraPath(str(t+1).zfill(5) +'_tomogram.vol')
                params += " -o " + self._getExtraPath(str(t+1).zfill(5) +'_tomogram.vol')
                params += " --geom " + self._getExtraPath(str(t+1).zfill(5) +"_"+str(i+1).zfill(5) +'_tomogram_map.xmd')
                params += " --ref " + self._getExtraPath(str(i + t*particlesPerTomogram +1).zfill(5) + '_deformed.vol')
                params += " --method copy "
                self.runJob('xmipp_tomo_map_back', params)

    def project_volumes(self):
        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            # If tomograms are selected, the volumes projected will be tomograms
            sizeX = self.tomoSizeX.get()
            sizeY = self.tomoSizeY.get()
            numberOfVolumes = self.numberOfTomograms.get()
            volumeName = "_tomogram.vol"
        else:
            # else, the deformed volumes are projected
            sizeX = self.volumeSize.get()
            sizeY = self.volumeSize.get()
            if self.modeRelationChoice.get() is MODE_RELATION_MESH:
                numberOfVolumes = self.meshRowPoints.get() * self.meshRowPoints.get()
            else:
                numberOfVolumes = self.numberOfVolumes.get()

            volumeName = "_deformed.vol"

        tiltStep = self.tiltStep.get()
        if self.missingWedgeChoice == MISSINGWEDGE_YES:
            tiltLow, tiltHigh = self.tiltLow.get(), self.tiltHigh.get()
        else:
            tiltLow, tiltHigh = -90, 90

        with open(self._getExtraPath('projection.param'), 'a') as file:
            file.write(
                "\n".join([
                    "# XMIPP_STAR_1 *",
                    "# Projection Parameters",
                    "data_noname",
                    "# X and Y projection dimensions [Xdim Ydim]",
                    "_projDimensions '%(sizeX)s %(sizeY)s'" % locals(),
                    "# Angle Set Source -----------------------------------------------------------",
                    "# tilt axis, direction defined by rot and tilt angles in degrees",
                    "_angleRot 90",
                    "_angleTilt 90",
                    "# tilt axis offset in pixels",
                    "_shiftX 0",
                    "_shiftY 0",
                    "_shiftZ 0",
                    "# Tilting description [tilt0 tiltF tiltStep] in degrees",
                    "_projTiltRange '%(tiltLow)s %(tiltHigh)s %(tiltStep)s'" % locals(),
                    "# Noise description ----------------------------------------------------------",
                    "#     applied to angles [noise (bias)]",
                    "_noiseAngles '0 0'",
                    "#     applied to pixels [noise (bias)]",
                    "_noisePixelLevel '0 0'",
                    "#     applied to particle center coordenates [noise (bias)]",
                    "_noiseParticleCoord '0 0'"]))
        for i in range(numberOfVolumes):
            params = " -i " +  self._getExtraPath(str(i + 1).zfill(5) + volumeName)
            params += " --oroot " + self._getExtraPath(str(i + 1).zfill(5) + '_projected')
            params += " --params " + self._getExtraPath('projection.param')
            self.runJob('xmipp_tomo_project', params)

    def apply_noise_and_ctf(self):

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            numberOfVolumes = self.numberOfTomograms.get()

        else:
            if self.modeRelationChoice.get() is MODE_RELATION_MESH:
                numberOfVolumes = self.meshRowPoints.get() * self.meshRowPoints.get()
            else:
                numberOfVolumes = self.numberOfVolumes.get()

        with open(self._getExtraPath('ctf.param'), 'a') as file:
            file.write(
                "\n".join([
                    "# XMIPP_STAR_1 *",
                    "data_noname",
                    "_ctfVoltage " + str(self.ctfVoltage.get()),
                    "_ctfSphericalAberration " + str(self.ctfSphericalAberration.get()),
                    "_ctfSamplingRate " + str(self.samplingRate.get()),
                    "_magnification " + str(self.ctfMagnification.get()),
                    "_ctfDefocusU " + str(self.ctfDefocusU.get()),
                    "_ctfDefocusV " + str(self.ctfDefocusV.get()),
                    "_ctfQ0 " + str(self.ctfQ0.get())]))

        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            params += " --ctf " + self._getExtraPath('ctf.param')
            paramsNoiseCTF = params+ " --after_ctf_noise --targetSNR " + str(self.targetSNR.get())
            self.runJob('xmipp_phantom_simulate_microscope', paramsNoiseCTF)

            # the metadata for the i_th stack is self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            MD_i = md.MetaData(self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel'))
            for objId in MD_i:
                img_name = MD_i.getValue(md.MDL_IMAGE, objId)
                params_j = " -i " + img_name + " -o " + img_name
                params_j += " --ctf " + self._getExtraPath('ctf.param')
                self.runJob('xmipp_ctf_phase_flip', params_j)

    def reconstruct(self):
        if self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get() * self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()

        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_reconstructed.vol')

            if self.reconstructionChoice == RECONSTRUCTION_FOURIER:
                self.runJob('xmipp_reconstruct_fourier', params)
            elif self.reconstructionChoice == RECONSTRUCTION_WBP:
                self.runJob('xmipp_reconstruct_wbp', params)

    def createOutputStep(self):
        # first making a metadata for only the subtomograms:
        out_mdfn = self._getExtraPath('subtomograms.xmd')
        pattern = '"' + self._getExtraPath() + '/*_reconstructed.vol"'
        command = '-p ' + pattern + ' -o ' + out_mdfn
        self.runJob('xmipp_metadata_selfile_create', command)
        # now creating the output set of volumes as output:
        partSet = self._createSetOfVolumes('subtomograms')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.samplingRate.get())
        self._defineOutputs(outputVolumes=partSet)

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
