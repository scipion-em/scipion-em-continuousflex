# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
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
from pyworkflow.utils import replaceBaseExt, replaceExt, getExt
from pyworkflow.utils import isPower2, getListFromRangeString
from pyworkflow.utils.path import copyFile, cleanPath, createLink
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from pwem.convert import cifToPdb
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
from pwem.utils import runProgram
import time
import glob
from joblib import dump
from math import cos, sin, pi
from continuousflex.protocols.convert import matrix2eulerAngles


NMA_ALIGNMENT_WAV = 0
NMA_ALIGNMENT_PROJ = 1

MODE_RELATION_LINEAR = 0
MODE_RELATION_3CLUSTERS = 1
MODE_RELATION_MESH = 2
MODE_RELATION_RANDOM = 3
MODE_RELATION_PARABOLA = 4

MISSINGWEDGE_YES = 0
MISSINGWEDGE_NO = 1

LOWPASS_YES = 0
LOWPASS_NO = 1

ROTATION_SHIFT_YES = 0
ROTATION_SHIFT_NO = 1

RECONSTRUCTION_FOURIER = 0
RECONSTRUCTION_WBP = 1

NOISE_CTF_YES = 0
NOISE_CTF_NO = 1

FULL_TOMOGRAM_YES= 0
FULL_TOMOGRAM_NO = 1

NMA_NO = 0
NMA_YES = 1

ROTATION_UNIFORM = 0
ROTATION_GAUSS = 1

class FlexProtSynthesizeSubtomo(ProtAnalysis3D):
    """ Protocol for synthesizing subtomograms. """
    _label = 'synthesize subtomograms'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('confVar', params.BooleanParam, default=NMA_YES,
                      label='Simulate conformational variability?',
                      help='If yes, you need to use normal mode analysis. If no, the given volume or atomic structure '
                           'will be used to simulate single particle subtomograms.')
        group1 = form.addGroup('Select an atomic structure OR an EM volume', condition='confVar==%d' % NMA_NO)
        group1.addParam('refAtomic', params.PointerParam ,pointerClass='AtomStruct', allowsNull=True, label='atomic structure')
        group1.addParam('refVolume', params.PointerParam ,pointerClass='Volume', allowsNull=True, label='EM volume')

        group2 = form.addGroup('conformational variability', condition='confVar==%d' % NMA_YES)
        group2.addParam('importPdbs', params.BooleanParam, default=False,
                        expertLevel=params.LEVEL_ADVANCED, label='Do you want to improt a set of heterogenous PDBs?',
                        help='if you keep this as No, you can use a set of normal modes to generate conformational '
                             'heterogeneity. Otherwise, set as Yes if you have a set of different PDBs that you want '
                             'to use to simulate subtomograms (for example a trajectory from molecular '
                             'dynamics simulation)')
        group2.addParam('pdbs_path', params.PathParam,
                      condition='importPdbs',
                      label="List of PDBs",
                      help='Use the file pattern as file location with /*.pdb')
        group2.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      allowsNull=True, condition='importPdbs==False',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        group2.addParam('modeList', NumericRangeParam,
                      label="Modes selection", condition='importPdbs==False',
                      allowsNull=True,
                      default='7-8',
                      help='Select the normal modes that will be used for image synthesis. \n'
                           'It is usually two modes that should be selected, unless if the relationship is linear or random.\n'
                           'You have several ways to specify the modes.\n'
                           ' Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        group2.addParam('modeRelationChoice', params.EnumParam, default=MODE_RELATION_LINEAR,
                      allowsNull=True, condition='importPdbs==False',
                      choices=['Linear relationship', '3 Clusters', 'Grid', 'Random', 'Upper half circle'],
                      label='Relationship between the modes',
                      help='linear relationship: all the selected modes will have equal amplitudes. \n'
                           '3 clusters: the volumes will be devided exactly into three classes.\n'
                           'Grid: the amplitudes will be in a grid shape (grid size is square of what grid step).\n'
                           'Random: all the amplitudes will be random in the given range.\n'
                           'Parabolic: The relationship between two modes will be upper half circle.')
        group2.addParam('centerPoint', params.IntParam, default=100,
                      allowsNull=True,
                      condition='importPdbs==False and modeRelationChoice==%d' % MODE_RELATION_3CLUSTERS,
                      label='Center point',
                      help='This number will be used to determine the distance between the clusters'
                           'center1 = (-center_point, 0)'
                           'center2 = (center_point, 0)'
                           'center3 = (0, center_point)')
        group2.addParam('modesAmplitudeRange', params.IntParam, default=150,
                        allowsNull=True,
                        condition='importPdbs==False and modeRelationChoice==%d or modeRelationChoice==%d or modeRelationChoice==%d'
                                  ' or modeRelationChoice==%d' % (MODE_RELATION_LINEAR, MODE_RELATION_MESH,
                                                                  MODE_RELATION_RANDOM, MODE_RELATION_PARABOLA),
                        label='Amplitude range N --> [-N, N]',
                        help='Choose the number N for which the generated normal mode amplitudes are in the range of [-N, N]')
        group2.addParam('meshRowPoints', params.IntParam, default=6,
                      allowsNull=True,
                      condition='importPdbs==False and modeRelationChoice==%d' % MODE_RELATION_MESH,
                      label='Grid number of steps',
                      help='This number will be the number of points in the row and the column (grid shape will be size*size)')
        form.addParam('numberOfVolumes', params.IntParam, default=36,
                      label='Number of volumes',
                      condition='importPdbs==False and modeRelationChoice!=%d'% MODE_RELATION_MESH,
                      help='Number of volumes that will be generated')
        form.addParam('samplingRate', params.FloatParam, default=2.2,
                      condition='confVar==%d or refAtomic!=None' % NMA_YES,
                      label='Sampling rate',
                      help='The sampling rate (voxel size in Angstroms)')
        form.addParam('volumeSize', params.IntParam, default=64,
                      condition='confVar==%d or refAtomic!=None' % NMA_YES,
                      label='Image size',
                      help='Volume size in voxels (all volumes will be cubes)')
        form.addParam('seedOption', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Random seed',
                      help='Keeping it as True means that different runs will generate different images in terms '
                           'of conforamtion and rigid-body parameters. If you set as False, then different runs will '
                           'have the same conformations and angles '
                           '(setting to False allows you to generate the same conformations and orientations with '
                           'different noise values).')

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
                      help='If not selected, noise and CTF will not be simulated. If selected, noise and CTF will be applied'
                           ' then the the projections are CTF phase inverted before reconstructed into a volume.')
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

        form.addSection(label='Low pass filtering')
        form.addParam('lowPassChoice', params.EnumParam, default=LOWPASS_NO,
                      choices=['Add extra dose accumulation', 'Stay with only CTF'],
                      label='Use low pass filtering',
                      help='The generated volumes will be low pass filtered before projection into a tilt series.'
                           ' This simulates extra distortions similar to dose accumulation.'
                           ' However, CTF will already have such an effect.')
        line = form.addLine('Frequency (normalized)',
                            condition='lowPassChoice==%d' % LOWPASS_YES,
                            help='The cufoff frequency and raised coside width of the low pass filter.'
                                 ' For details: see "xmipp_transform_filter --fourier low_pass"')
        line.addParam('w1', params.FloatParam, default=0.25,
                      condition='lowPassChoice==%d' % LOWPASS_YES,
                      label='Cutoff frequency (0 -> 0.5)')
        line.addParam('raisedw', params.FloatParam, default=0.02,
                      condition='lowPassChoice==%d' % LOWPASS_YES,
                      label='Raised cosine width')

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

        form.addParam('shiftx', params.EnumParam, label='Shift-X distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        groupx_uni = form.addGroup('Shift-X Uniform distribution', condition='rotationShiftChoice==%d and shiftx==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        groupx_uni.addParam('LowX', params.FloatParam, default=-5.0,
                      label='Lower value')
        groupx_uni.addParam('HighX', params.FloatParam, default=5.0,
                      label='Higher value')
        groupx_gas = form.addGroup('Shift-X Gaussian distribution', condition='rotationShiftChoice==%d and shiftx==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        groupx_gas.addParam('MeanX', params.FloatParam, default=0.0,
                      label='Mean value for the Gaussian distribution')
        groupx_gas.addParam('StdX', params.FloatParam, default=2.0,
                      label='Standard deviation for the Gaussian distribution')

        form.addParam('shifty', params.EnumParam, label='Shift-Y distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        group = form.addGroup('Shift-Y Uniform distribution', condition='rotationShiftChoice==%d and shifty==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        group.addParam('LowY', params.FloatParam, default=-5.0,
                      label='Lower value')
        group.addParam('HighY', params.FloatParam, default=5.0,
                      label='Higher value')
        group = form.addGroup('Shift-Y Gaussian distribution', condition='rotationShiftChoice==%d and shifty==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        group.addParam('MeanY', params.FloatParam, default=0.0,
                      label='Mean value for the Gaussian distribution')
        group.addParam('StdY', params.FloatParam, default=2.0,
                      label='Standard deviation for the Gaussian distribution')

        form.addParam('shiftz', params.EnumParam, label='Shift-Z distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        group = form.addGroup('Shift-Z Uniform distribution', condition='rotationShiftChoice==%d and shiftz==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        group.addParam('LowZ', params.FloatParam, default=-5.0,
                      label='Lower value')
        group.addParam('HighZ', params.FloatParam, default=5.0,
                      label='Higher value')
        group = form.addGroup('Shift-Z Gaussian distribution', condition='rotationShiftChoice==%d and shiftz==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        group.addParam('MeanZ', params.FloatParam, default=0.0,
                      label='Mean value for the Gaussian distribution')
        group.addParam('StdZ', params.FloatParam, default=2.0,
                      label='Standard deviation for the Gaussian distribution')

        form.addParam('rot', params.EnumParam, label='Rot angle distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        group = form.addGroup('Rot angle Uniform distribution', condition='rotationShiftChoice==%d and rot==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        group.addParam('LowRot', params.FloatParam, default=0.0,
                      label='Lower value')
        group.addParam('HighRot', params.FloatParam, default=360.0,
                      label='Higher value')
        group = form.addGroup('Rot angle Gaussian distribution', condition='rotationShiftChoice==%d and rot==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        group.addParam('MeanRot', params.FloatParam, default=180.0,
                      label='Mean value for the Gaussian distribution')
        group.addParam('StdRot', params.FloatParam, default=90.0,
                      label='Standard deviation for the Gaussian distribution')

        form.addParam('tilt', params.EnumParam, label='Tilt angle distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        group = form.addGroup('Tilt angle Uniform distribution', condition='rotationShiftChoice==%d and tilt==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        group.addParam('LowTilt', params.FloatParam, default=0.0,
                      label='Lower value')
        group.addParam('HighTilt', params.FloatParam, default=180.0,
                      label='Higher value')
        group = form.addGroup('Tilt angle Gaussian distribution', condition='rotationShiftChoice==%d and tilt==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        group.addParam('MeanTilt', params.FloatParam, default=90.0,
                      label='Mean value for the Gaussian distribution')
        group.addParam('StdTilt', params.FloatParam, default=45.0,
                      label='Standard deviation for the Gaussian distribution')

        form.addParam('psi', params.EnumParam, label='Psi angle distribution',
                      condition='rotationShiftChoice==%d' % ROTATION_SHIFT_YES,
                      choices=['Uniform distribution', 'Gaussian distribution'],
                      default=ROTATION_UNIFORM,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help='The distribution of random values')
        group = form.addGroup('Psi angle Uniform distribution', condition='rotationShiftChoice==%d and psi==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_UNIFORM))
        group.addParam('LowPsi', params.FloatParam, default=0.0,
                      label='Lower value')
        group.addParam('HighPsi', params.FloatParam, default=360.0,
                      label='Higher value')
        group = form.addGroup('Psi angle Gaussian distribution', condition='rotationShiftChoice==%d and psi==%d'
                                                                             % (ROTATION_SHIFT_YES,ROTATION_GAUSS))
        group.addParam('MeanPsi', params.FloatParam, default=180.0,
                      label='Mean value for the Gaussian distribution')
        group.addParam('StdPsi', params.FloatParam, default=90.0,
                      label='Standard deviation for the Gaussian distribution')

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
        if(self.seedOption.get()):
            np.random.seed(int(time.time()))
        else:
            np.random.seed(0)
        if(self.confVar.get()==NMA_YES):
            if(self.importPdbs.get()):
                self._insertFunctionStep("copy_deformations")
            else:
                self._insertFunctionStep("generate_deformations")
            self._insertFunctionStep("generate_volume_from_pdb")
        else:
            self._insertFunctionStep("generate_copies_of_volume")
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
        # Find the right PDB file to use for data synthesis
        pdb_name1 = os.path.dirname(self.inputModes.get().getFileName()) + '/atoms.pdb'
        pdb_name2 = os.path.dirname(self.inputModes.get().getFileName()) + '/pseudoatoms.pdb'
        if os.path.exists(pdb_name1):
            fnPDB = pdb_name1
        else:
            fnPDB = pdb_name2
        # fnPDB = self.inputModes.get().getPdb().getFileName()
        # use the input relationship between the modes to generate normal mode amplitudes metadata
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
        numberOfVolumes = self.get_number_of_volumes()
        # if self.modeRelationChoice.get() is MODE_RELATION_MESH:
        #     numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        # else:
        #     numberOfVolumes = self.numberOfVolumes.get()
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
            elif self.modeRelationChoice == MODE_RELATION_PARABOLA:
                amplitude=self.modesAmplitudeRange.get()
                rv = np.random.uniform(0,1)
                point = (amplitude*cos(rv*pi), amplitude*sin(rv*pi))
                deformations[modeSelection-1] = point

            # we won't keep the first 6 modes
            deformations = deformations[6:]

            params = " --pdb " + fnPDB
            params+= " --nma " + fnModeList
            params+= " -o " + self._getExtraPath(str(i+1).zfill(5)+'_df.pdb')
            params+= " --deformations " + ' '.join(str(i) for i in deformations)
            runProgram('xmipp_pdb_nma_deform', params)

            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1).zfill(5)+'_subtomogram'+'.vol'), subtomogramMD.addObject())
            subtomogramMD.setValue(md.MDL_NMA, list(deformations), i+1)

        subtomogramMD.write(deformationFile)


    def copy_deformations(self):
        pdbs_list = [f for f in glob.glob(self.pdbs_path.get())]
        # print(pdbs_list)
        # saving the list
        dump(pdbs_list, self._getExtraPath('pdb_list.pkl'))
        subtomogramMD = md.MetaData()
        print(self.get_number_of_volumes())
        i = 0
        for pdbfn in pdbs_list:
            i += 1
            createLink(pdbfn, self._getExtraPath(str(i).zfill(5)+'_df.pdb'))
            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i).zfill(5)+'_subtomogram'+'.vol'),
                                   subtomogramMD.addObject())
        subtomogramMD.write(self._getExtraPath('GroundTruth.xmd'))


    def generate_volume_from_pdb(self):
        numberOfVolumes = self.get_number_of_volumes()

        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_df.pdb')
            params += " --sampling " + str(self.samplingRate.get())
            params += " --size " + str(self.volumeSize.get())
            params += " -v 0 --centerPDB "
            runProgram('xmipp_volume_from_pdb', params)

        if self.lowPassChoice.get() is LOWPASS_YES:
            cutoff = self.w1.get()
            raisedw = self.raisedw.get()
            for i in range(numberOfVolumes):
                params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_df.vol')
                params += " --fourier low_pass " + str(cutoff) + ' ' + str(raisedw)
                runProgram('xmipp_transform_filter', params)

    def generate_rotation_and_shift(self):
        subtomogramMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        numberOfVolumes = self.get_number_of_volumes()

        for i in range(numberOfVolumes):
            if (self.shiftx.get()==ROTATION_UNIFORM):
                shift_x1 = np.random.uniform(self.LowX.get(), self.HighX.get())
            else:
                shift_x1 = np.random.normal(self.MeanX.get(), self.StdX.get())
            if (self.shifty.get()==ROTATION_UNIFORM):
                shift_y1 = np.random.uniform(self.LowY.get(), self.HighY.get())
            else:
                shift_y1 = np.random.normal(self.MeanY.get(), self.StdY.get())
            if (self.shiftz.get()==ROTATION_UNIFORM):
                shift_z1 = np.random.uniform(self.LowZ.get(), self.HighZ.get())
            else:
                shift_z1 = np.random.normal(self.MeanZ.get(), self.StdZ.get())
            if (self.rot.get()==ROTATION_UNIFORM):
                rot1 = np.random.uniform(self.LowRot.get(), self.HighRot.get())
            else:
                rot1 = np.random.normal(self.MeanRot.get(), self.StdRot.get())
            if (self.tilt.get()==ROTATION_UNIFORM):
                tilt1 = np.random.uniform(self.LowTilt.get(), self.HighTilt.get())
            else:
                tilt1 = np.random.normal(self.MeanTilt.get(), self.StdTilt.get())
            if (self.psi.get()==ROTATION_UNIFORM):
                psi1 = np.random.uniform(self.LowPsi.get(), self.HighPsi.get())
            else:
                psi1 = np.random.normal(self.MeanPsi.get(), self.StdPsi.get())

            # uniform over the sphere
            if (self.psi.get() == ROTATION_UNIFORM) and\
                (self.tilt.get()==ROTATION_UNIFORM) and \
                    (self.rot.get()==ROTATION_UNIFORM) and \
                    self.LowRot.get() == 0.0 and self.HighRot.get() == 360.0 and \
                    self.LowTilt.get() == 0.0 and self.HighTilt.get() == 180.0 and \
                    self.LowPsi.get() == 0.0 and self.HighPsi.get() == 360.0:
                x1,x2,x3 = np.random.uniform(0,1,3)
                R = np.array([
                    [np.cos(2*np.pi*x1), np.sin(2*np.pi*x1), 0],
                    [-np.sin(2*np.pi*x1), np.cos(2*np.pi*x1), 0],
                    [0, 0, 1]
                ])
                v = np.array([[np.cos(2*np.pi*x2)*np.sqrt(x3),
                              np.sin(2*np.pi*x2)*np.sqrt(x3),
                              np.sqrt(1-x3)]])
                H = np.eye(3) - 2*np.dot(v.T,v)
                M = -np.dot(H,R)
                trans_mat = np.zeros((4,4))
                trans_mat[:3,:3] = M
                rot1,tilt1,psi1,_,_,_ = matrix2eulerAngles(trans_mat)

            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_df.vol')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_df.vol')
            params += " --rotate_volume euler " + str(rot1) + ' ' + str(tilt1) + ' ' + str(psi1)
            params += " --shift " + str(shift_x1) + ' ' + str(shift_y1) + ' ' + str(shift_z1)
            params += " --dont_wrap "
            runProgram('xmipp_transform_geometry', params)

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
            runProgram('xmipp_phantom_create', params)

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
        numberOfVolumes=self.get_number_of_volumes()

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
                params += " --ref " + self._getExtraPath(str(i + t*particlesPerTomogram +1).zfill(5) + '_df.vol')
                params += " --method copy "
                runProgram('xmipp_tomo_map_back', params)

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
            numberOfVolumes = self.get_number_of_volumes()

            volumeName = "_df.vol"

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
            runProgram('xmipp_tomo_project', params)

    def apply_noise_and_ctf(self):

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            numberOfVolumes = self.numberOfTomograms.get()

        else:
            numberOfVolumes = self.get_number_of_volumes()

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
            runProgram('xmipp_phantom_simulate_microscope', paramsNoiseCTF)

            # the metadata for the i_th stack is self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            MD_i = md.MetaData(self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel'))
            for objId in MD_i:
                img_name = MD_i.getValue(md.MDL_IMAGE, objId)
                params_j = " -i " + img_name + " -o " + img_name
                params_j += " --ctf " + self._getExtraPath('ctf.param')
                runProgram('xmipp_ctf_phase_flip', params_j)

    def reconstruct(self):
        numberOfVolumes = self.get_number_of_volumes()
        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_subtomogram.vol')

            if self.reconstructionChoice == RECONSTRUCTION_FOURIER:
                runProgram('xmipp_reconstruct_fourier', params)
            elif self.reconstructionChoice == RECONSTRUCTION_WBP:
                runProgram('xmipp_reconstruct_wbp', params)

    def generate_copies_of_volume(self):
        fn_volume = self._getExtraPath('reference')
        if(self.refAtomic.get()):
            pdbFn = self.refAtomic.get().getFileName()
            if getExt(pdbFn) == ".cif":
                pdbFn2 = replaceBaseExt(pdbFn, 'pdb')
                cifToPdb(pdbFn, pdbFn2)
                pdbFn = pdbFn2
            params = " -i " + pdbFn
            params += " -o " + fn_volume
            params += " --sampling " + str(self.samplingRate.get())
            params += " --size " + str(self.volumeSize.get())
            params += " -v 0 --centerPDB "
            runProgram('xmipp_volume_from_pdb', params)
        else:
            # Convert to spider format in case it is MRC
            params = "-i " + self.refVolume.get().getFileName()
            params += " -o " + fn_volume + ".vol --type vol"
            runProgram('xmipp_image_convert', params)
            # print(self.refVolume.get().getFileName())
        numberOfVolumes = self.get_number_of_volumes()
        deformationFile = self._getExtraPath('GroundTruth.xmd')
        imagesMD = md.MetaData()
        for i in range(numberOfVolumes):
            Vol_i = self._getExtraPath(str(i + 1).zfill(5) + '_df.vol')
            copyFile(fn_volume+'.vol',Vol_i)
            imagesMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i + 1).zfill(5) + '_subtomogram' + '.spi'),
                                   imagesMD.addObject())

        imagesMD.write(deformationFile)


    def createOutputStep(self):
        # first making a metadata for only the subtomograms:
        out_mdfn = self._getExtraPath('subtomograms.xmd')
        pattern = '"' + self._getExtraPath() + '/*_subtomogram.vol"'
        command = '-p ' + pattern + ' -o ' + out_mdfn
        runProgram('xmipp_metadata_selfile_create', command)
        # now creating the output set of volumes as output:
        partSet = self._createSetOfVolumes('subtomograms')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.samplingRate.get())
        self._defineOutputs(outputVolumes=partSet)


    def get_number_of_volumes(self):
        if(self.importPdbs.get()):
            numberOfVolumes = len(glob.glob(self.pdbs_path.get()))
        elif self.modeRelationChoice.get() is MODE_RELATION_MESH:
            numberOfVolumes = self.meshRowPoints.get()*self.meshRowPoints.get()
        else:
            numberOfVolumes = self.numberOfVolumes.get()
        return numberOfVolumes

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

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022continuousflex','harastani2020hybrid']

    def _methods(self):
        pass