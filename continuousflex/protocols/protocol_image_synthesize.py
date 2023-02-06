# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
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

import os
import time

import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
import xmipp3
from pwem.protocols import ProtAnalysis3D
from pwem.utils import runProgram
from pyworkflow.protocol.params import NumericRangeParam
from pyworkflow.utils import getListFromRangeString
from pyworkflow.utils import replaceExt, getExt, replaceBaseExt
from pwem.convert import cifToPdb
from xmipp3.convert import (writeSetOfParticles)
from pyworkflow.utils.path import copyFile, createLink
import numpy as np
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

LOWPASS_YES = 0
LOWPASS_NO = 1

ROTATION_SHIFT_YES = 0
ROTATION_SHIFT_NO = 1

RECONSTRUCTION_FOURIER = 0
RECONSTRUCTION_WBP = 1

NOISE_CTF_YES = 0
NOISE_CTF_NO = 1

NMA_NO = 0
NMA_YES = 1

ROTATION_UNIFORM = 0
ROTATION_GAUSS = 1


class FlexProtSynthesizeImages(ProtAnalysis3D):
    """ Protocol for synthesizing images. """
    _label = 'synthesize images'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('confVar', params.BooleanParam, default=NMA_YES,
                      label='Simulate conformational variability?',
                      help='If yes, you need to use normal mode analysis. If no, the given volume or atomic structure '
                           'will be projected into simulated particle images.')
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
                      label='Number of images',
                      condition='importPdbs==False and modeRelationChoice!=%d'% MODE_RELATION_MESH,
                      help='Number of images that will be generated')
        form.addParam('samplingRate', params.FloatParam, default=2.2,
                      condition='confVar==%d or refAtomic!=None' % NMA_YES,
                      label='Sampling rate',
                      help='The sampling rate (pixel size in Angstroms)')
        form.addParam('volumeSize', params.IntParam, default=64,
                      condition='confVar==%d or refAtomic!=None' % NMA_YES,
                      label='Image size',
                      help='Image size in voxels (all volumes will be cubes)')
        form.addParam('seedOption', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Random seed',
                      help='Keeping it as True means that different runs will generate different images in terms '
                           'of conforamtion and rigid-body parameters. If you set as False, then different runs will '
                           'have the same conformations and angles '
                           '(setting to False allows you to generate the same conformations and orientations with '
                           'different noise values).')

        form.addSection(label='Noise and CTF')
        form.addParam('noiseCTFChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Apply Noise and CTF',
                      help='If not selected, noise and CTF will not be simulated. If selected, noise and CTF will be applied'
                           ' then the the projections are CTF phase inverted.')
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
                      help='The generated volumes will be low pass filtered before projection to images.'
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

        form.addSection('Rigid body variability')
        form.addParam('rotationShiftChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Simulate Rotations and Shifts',
                      help='If yes, the images will be rotated and shifted randomly in the range,'
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
            if self.rotationShiftChoice == ROTATION_SHIFT_YES:
                self._insertFunctionStep("generate_rotation_and_shift")
            if self.rotationShiftChoice == ROTATION_SHIFT_NO:
                self._insertFunctionStep("generate_zero_rotation_and_shift")
            self._insertFunctionStep("project_volumes")
            if self.noiseCTFChoice == NOISE_CTF_YES:
                self._insertFunctionStep("apply_noise_and_ctf")
        else:
            self._insertFunctionStep("generate_links_to_volume")
            if self.rotationShiftChoice == ROTATION_SHIFT_YES:
                self._insertFunctionStep("generate_rotation_and_shift")
            if self.rotationShiftChoice == ROTATION_SHIFT_NO:
                self._insertFunctionStep("generate_zero_rotation_and_shift")
            self._insertFunctionStep("project_volumes")
            if self.noiseCTFChoice == NOISE_CTF_YES:
                self._insertFunctionStep("apply_noise_and_ctf")
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

        # iterate over the number of outputs (if mesh, this has to be calculated)
        numberOfVolumes = self.get_number_of_volumes()

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
            elif self.modeRelationChoice == MODE_RELATION_PARABOLA:
                amplitude=self.modesAmplitudeRange.get()
                rv = np.random.uniform(0,1)
                point = (amplitude*cos(rv*pi), amplitude*sin(rv*pi))
                deformations[modeSelection-1] = point

            elif self.modeRelationChoice == MODE_RELATION_MESH:
                new_point=(mode7_samples[i],mode8_samples[i])
                deformations[modeSelection - 1] = new_point

            # we won't keep the first 6 modes
            deformations = deformations[6:]

            # params = " --pdb " + fnPDB
            # params+= " --nma " + fnModeList
            # params+= " -o " + self._getExtraPath(str(i+1).zfill(5)+'_df.pdb')
            # params+= " --deformations " + ' '.join(str(i) for i in deformations)
            # runProgram('xmipp_pdb_nma_deform', params)
            self.nma_deform_pdb(fnPDB,fnModeList,self._getExtraPath(str(i+1).zfill(5)+'_df.pdb'),deformations)

            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1).zfill(5)+'_projected'+'.spi'), subtomogramMD.addObject())
            subtomogramMD.setValue(md.MDL_NMA, list(deformations), i+1)

        subtomogramMD.write(deformationFile)
    def copy_deformations(self):
        pdbs_list = [f for f in glob.glob(self.pdbs_path.get())]
        pdbs_list.sort()
        # print(pdbs_list)
        # saving the list
        dump(pdbs_list, self._getExtraPath('pdb_list.pkl'))
        subtomogramMD = md.MetaData()
        i = 0
        for pdbfn in pdbs_list:
            i += 1
            createLink(pdbfn, self._getExtraPath(str(i).zfill(5)+'_df.pdb'))
            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i).zfill(5)+'_projected'+'.spi'),
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

            subtomogramMD.setValue(md.MDL_SHIFT_X, shift_x1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Y, shift_y1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_ROT, rot1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_TILT, tilt1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_PSI, psi1, i + 1)
        subtomogramMD.write(self._getExtraPath('GroundTruth.xmd'))

    def generate_zero_rotation_and_shift(self):
        subtomogramMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        numberOfVolumes = self.get_number_of_volumes()

        for i in range(numberOfVolumes):
            rot1 = 0.0
            tilt1 = 0.0
            psi1 = 0.0
            shift_x1 = 0.0
            shift_y1 = 0.0
            subtomogramMD.setValue(md.MDL_SHIFT_X, shift_x1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Y, shift_y1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_ROT, rot1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_TILT, tilt1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_PSI, psi1, i + 1)
        subtomogramMD.write(self._getExtraPath('GroundTruth.xmd'))


    def project_volumes(self):
        numberOfVolumes = self.get_number_of_volumes()

        sizeX = self.volumeSize.get()
        sizeY = self.volumeSize.get()
        volumeName = "_df.vol"
        subtomogramMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        for i in range(numberOfVolumes):
            rot = subtomogramMD.getValue(md.MDL_ANGLE_ROT, i+1)
            tilt = subtomogramMD.getValue(md.MDL_ANGLE_TILT, i+1)
            psi = subtomogramMD.getValue(md.MDL_ANGLE_PSI, i+1)
            x = subtomogramMD.getValue(md.MDL_SHIFT_X, i+1)
            y = subtomogramMD.getValue(md.MDL_SHIFT_Y, i+1)

            params = " -i " +  self._getExtraPath(str(i + 1).zfill(5) + volumeName)
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.spi')
            params += " --angles " + str(rot) + ' ' + str(tilt) + ' ' + str(psi) + ' ' + str(x) + ' ' + str(y)
            runProgram('xmipp_phantom_project', params)

    def apply_noise_and_ctf(self):
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
        #
        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.spi')
            params += " --ctf " + self._getExtraPath('ctf.param')
            paramsNoiseCTF = params+ " --after_ctf_noise --targetSNR " + str(self.targetSNR.get())
            runProgram('xmipp_phantom_simulate_microscope', paramsNoiseCTF)

            # Phase flip:
            img_name = self._getExtraPath(str(i + 1).zfill(5)) + '_projected.spi'
            params_j = " -i " + img_name + " -o " + img_name
            params_j += " --ctf " + self._getExtraPath('ctf.param')
            runProgram('xmipp_ctf_phase_flip', params_j)

    def nma_deform_pdb(self, fnPDB, fnModeList, fnOut, deformList):
        def readPDB(fnIn):
            with open(fnIn) as f:
                lines = f.readlines()
            newlines = []
            for line in lines:
                if line.startswith("ATOM "):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        newline = [x, y, z]
                        newlines.append(newline)
                    except:
                        pass
            return newlines

        def readModes(fnIn):
            modesMD = md.MetaData(fnIn)
            vectors = []
            for objId in modesMD:
                vecFn = modesMD.getValue(md.MDL_NMA_MODEFILE, objId)
                vec = np.loadtxt(vecFn)
                vectors.append(vec)
            return vectors

        def savePDB(list, fn, fn_original):
            with open(fn_original) as f:
                lines = f.readlines()
            newLines = []
            i = 0
            for line in lines:
                if line.startswith("ATOM "):
                    try:
                        x = list[i][0]
                        y = list[i][1]
                        z = list[i][2]
                        newLine = line[0:30] + "%8.3f%8.3f%8.3f" % (x, y, z) + line[54:]
                        i += 1
                    except:
                        pass
                else:
                    newLine = line
                newLines.append(newLine)
            with open(fn, mode='w') as f:
                f.writelines(newLines)
            pass

        pdb_array = np.array(readPDB(fnPDB))
        modes = readModes(fnModeList)
        for i in range(len(deformList)):
            pdb_array += deformList[i] * modes[7-1+i]
        savePDB(pdb_array, fnOut, fnPDB)

    def generate_links_to_volume(self):
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
            createLink(fn_volume+'.vol',Vol_i)
            imagesMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i + 1).zfill(5) + '_projected' + '.spi'),
                                   imagesMD.addObject())

        imagesMD.write(deformationFile)

    def createOutputStep(self):
        # first making a metadata for only the images:
        out_mdfn = self._getExtraPath('images.xmd')
        pattern = '"' + self._getExtraPath() + '/*_projected.spi"'
        command = '-p ' + pattern + ' -o ' + out_mdfn
        runProgram('xmipp_metadata_selfile_create', command)
        # now creating the output set of images as output:
        partSet = self._createSetOfParticles('images')
        xmipp3.convert.readSetOfParticles(self._getExtraPath('GroundTruth.xmd'), partSet)
        if (self.refVolume.get()):
            sr = self.refVolume.get().getSamplingRate()
        else:
            sr = self.samplingRate.get()
        partSet.setSamplingRate(sr)
        self._defineOutputs(outputImages=partSet)



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
        return ['harastani2022continuousflex']

    def _methods(self):
        pass