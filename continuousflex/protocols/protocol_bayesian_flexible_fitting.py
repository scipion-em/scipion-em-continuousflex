# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
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

import pyworkflow.protocol.params as params
from pyworkflow.utils import getListFromRangeString
from pwem.protocols import ProtAnalysis3D
from pwem.objects.data import AtomStruct
# from pyworkflow.utils import replaceExt
# import pwem.emlib.metadata as md



import sys
sys.path.append('/home/guest/PycharmProjects/bayesian-md-nma')
import src.molecule
import src.density
import src.functions
from src.viewers import chimera_fit_viewer, chimera_molecule_viewer
from src.flexible_fitting import FlexibleFitting, multiple_fitting
import numpy as np

ATOMICMODEL_CARBONALPHA=0
ATOMICMODEL_BACKBONE=1
ATOMICMODEL_ALLATOMS=2

class FlexProtBayesianFlexibleFitting(ProtAnalysis3D):
    """ Protocol for Bayesian Flexible Fitting. """
    _label = 'bayesian flexible fitting'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ########################### Input ############################################
        form.addSection(label='Input')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB", important=True,
                      help='Select the reference PDB')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select the set of volumes that will be analyzed using normal modes.')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', params.NumericRangeParam,
                      label="Modes selection",
                      default='7-8',
                      help='Select the normal modes that will be used for image analysis. \n'
                           'It is usually two modes that should be selected, unless if the relationship is linear or random.\n'
                           'You have several ways to specify the modes.\n'
                           ' Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('n_proc', params.IntParam, default=4, label='Number of processors',
                      help="TODO")
        form.addParam('verboseLevel', params.IntParam, default=0, label='Verbose level',
                      help="TODO", expertLevel=params.LEVEL_ADVANCED)

        ########################### Force field ############################################
        form.addSection(label='Atomic Model')
        form.addParam('atomicModel', params.EnumParam, label="Atomic model", default=ATOMICMODEL_ALLATOMS,
                      choices=['Carbon Alpha', 'Backbone', 'All atoms'],
                      help="TODO")
        form.addParam('inputPSF', params.FileParam, label="Protein Structure File (PSF)", condition='atomicModel>0',
                      help='Generated with generatePSF protocol (.psf)')
        form.addParam('inputPRM', params.FileParam, label="Parameter File (PRM)", condition='atomicModel>0',
                      help='CHARMM force field parameter file (.prm)')

        ########################### Density ############################################
        form.addSection(label='Density')
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="TODO")
        form.addParam('gauss_sigma', params.FloatParam, default=2.0, label='3D Gaussians standard deviation',
                      help="TODO")
        form.addParam('gauss_threshold', params.IntParam, default=5, label='3D Gaussians cutoff',
                      help="TODO")


        ########################### Bayesian model ############################################
        form.addSection(label='Bayesian Model')
        form.addParam('biaisingFactor', params.FloatParam, default=100.0, label='Biasing potential factor',
                      help="TODO")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="TODO")
        form.addParam('fitLocal', params.BooleanParam, label="Fit local dynamics",
                      default=True,
                      help="TODO")
        form.addParam('dtLocal', params.FloatParam, default=1,
                      label='Local dynamics time step (fs)',
                      condition='fitLocal==True', expertLevel=params.LEVEL_ADVANCED,
                      help="TODO")
        form.addParam('fitGlobal', params.BooleanParam, label="Fit global dynamics",
                      default=True,
                      help="TODO")
        form.addParam('dtGlobal', params.FloatParam, default=0.05,
                      label='Global dynamics time step',
                      condition='fitGlobal==True', expertLevel=params.LEVEL_ADVANCED,
                      help="TODO")
        form.addParam('fitRotation', params.BooleanParam, label="Fit rotations",
                      default=False,
                      help="TODO")
        form.addParam('dtRotation', params.FloatParam, default=0.00005,
                      label='Rotations time step',
                      condition='fitRotation==True', expertLevel=params.LEVEL_ADVANCED,
                      help="TODO")
        form.addParam('fitShift', params.BooleanParam, label="Fit translations",
                      default=False,
                      help="TODO")
        form.addParam('dtShift', params.FloatParam, default=0.00005,
                      label='Translations time step',
                      condition='fitShift==True', expertLevel=params.LEVEL_ADVANCED,
                      help="TODO")

        ########################### HMC Parameters ############################################
        form.addSection(label='Fitting Parameters')
        form.addParam('n_iter', params.IntParam, default=50,  label='Number of iterations',
                      help="TODO")
        form.addParam('n_warmup', params.IntParam, default=25, label='Number of warmup ',
                      help="TODO")
        form.addParam('n_chain', params.IntParam, default=4, label='Number of chains',
                      help="TODO")
        form.addParam('n_step', params.IntParam, default=20, label='Steps per iteration',
                      help="TODO")

        # --------------------------- INSERT steps functions --------------------------------------------


    def _insertAllSteps(self):
        self.createInitialStructure()
        self.createTargetDensity()
        self.createFittingParameters()
        chimera_fit_viewer(mol=self.initStructure, target=self.targetDensities[0])
        self.runHMC()
        self.createOutputStep()

    def createInitialStructure(self):
        # Read PDB
        fnPDB = self.inputPDB.get().getFileName()
        self.initStructure = src.molecule.Molecule(pdb_file=fnPDB)

        # Add selected modes
        fnModes = [i.getModeFile() for i in self.inputModes.get()]
        modeSelection = np.array(getListFromRangeString(self.modeList.get()))
        self.initStructure.set_normalModeVec(files=fnModes, selection=modeSelection)

        # Set Molecule Force field
        if self.atomicModel.get() == ATOMICMODEL_CARBONALPHA:
            # If coarse grained, use default Carbon alpha force field values
            self.initStructure.allatoms2carbonalpha()
            self.initStructure.set_forcefield()
        else:
            # Else read PSF and PRM files
            fnPSF = self.inputPSF.get()
            fnPRM = self.inputPRM.get()
            self.initStructure.set_forcefield(psf_file=fnPSF, prm_file=fnPRM)
            if self.atomicModel.get() == ATOMICMODEL_BACKBONE:
                self.initStructure.allatoms2backbone()

    def createTargetDensity(self):
        # Read Volumes
        fnVolumes = [i.getFileName() for i in self.inputVolumes.get()]
        self.targetDensities = []
        for i in fnVolumes:
            targetDensity = src.density.Volume.from_file(file=i, sigma=self.gauss_sigma.get(),
                              threshold=self.gauss_threshold.get(), voxel_size=self.voxel_size.get())

            # Rescale the volumes to the initial structure
            initDensity = src.density.Volume.from_coords(coord=self.initStructure.coords, size=targetDensity.size,
                                                         sigma = targetDensity.sigma, threshold=targetDensity.threshold,
                                                         voxel_size=targetDensity.voxel_size)
            targetDensity.rescale(initDensity)
            self.targetDensities.append(targetDensity)

    def createFittingParameters(self):
        self.fittingParams = {
            "initial_biasing_factor": self.biaisingFactor.get(),
            "n_step": self.n_step.get(),
            "n_iter" : self.n_iter.get(),
            "n_warmup" : self.n_warmup.get(),
            "criterion": False,
            "temperature" :  self.temperature.get()
        }
        self.fittingVars =[]
        if self.fitLocal.get():
            self.fittingVars.append("local")
            self.fittingParams["local_dt"] = self.dtLocal.get() * 1e-15
        if self.fitGlobal.get():
            self.fittingVars.append("global")
            self.fittingParams["global_dt"] = self.dtGlobal.get()
        if self.fitRotation.get():
            self.fittingVars.append("rotation")
            self.fittingParams["rotation_dt"] = self.dtRotation.get()
        if self.fitShift.get():
            self.fittingVars.append("shift")
            self.fittingParams["shift_dt"] = self.dtShift.get()

    def runHMC(self):
        # Run parallel HMC fitting
        models=[]
        for t in self.targetDensities:
            models.append(FlexibleFitting(init=self.initStructure, target=t, vars=self.fittingVars,n_chain=self.n_chain.get(),
                        params=self.fittingParams, verbose=self.verboseLevel.get()))

        self.fittingHMC = multiple_fitting(models = models,n_chain=self.n_chain.get(), n_proc=self.n_proc.get())

    def createOutputStep(self):
        for i in range(len(self.fittingHMC)) :
            self.fittingHMC[i].res["mol"].save_pdb(file=self._getExtraPath(str(i).zfill(5)+'_fitted.pdb'))
        # pdb = AtomStruct(self._getExtraPath('fitted.pdb'), pseudoatoms=False)
        # self._defineOutputs(outputPdb=pdb)


    # --------------------------- STEPS functions --------------------------------------------
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------