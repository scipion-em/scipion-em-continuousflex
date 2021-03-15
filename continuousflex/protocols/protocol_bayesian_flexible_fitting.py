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

import continuousflex.protocols.src.molecule
import continuousflex.protocols.src.density
import continuousflex.protocols.src.functions
from continuousflex.protocols.src.viewers import chimera_fit_viewer


import numpy as np
import os

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
        form.addSection(label='Force Field')
        form.addParam('biaisingFactor', params.FloatParam, default=1.0, label='Biasing potential factor',
                      help="TODO")
        form.addParam('energyFactor', params.FloatParam, default=1.0, label='Potential energy factor',
                      help="TODO")
        form.addParam('coarseGrained', params.BooleanParam, label="Coarse-grained model",
                      default=True,
                      help="TODO")
        form.addParam('inputPSF', params.FileParam, label="Protein Structure File (PSF)", condition='coarseGrained==False',
                      help='Generated with generatePSF protocol (.psf)')
        form.addParam('inputPRM', params.FileParam, label="Parameter File (PRM)", condition='coarseGrained==False',
                      help='CHARMM force field parameter file (.prm)')

        ########################### Volume parameters ############################################
        form.addSection(label='Volume parameters')
        form.addParam('gauss_sigma', params.FloatParam, default=2.0, label='3D Gaussians standard deviation',
                      help="TODO")
        form.addParam('gauss_threshold', params.IntParam, default=5, label='3D Gaussians cutoff',
                      help="TODO")


        ########################### Bayesian model ############################################
        form.addSection(label='Bayesian Model')
        form.addParam('fitLocal', params.BooleanParam, label="Fit local dynamics",
                      default=True,
                      help="TODO")
        form.addParam('dtLocal', params.FloatParam, default=0.005,
                      label='Local dynamics time step',
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
        self.runHMC()
        self.createOutputStep()

    def createInitialStructure(self):
        # Read PDB
        fnPDB = self.inputPDB.get().getFileName()
        self.initStructure = continuousflex.protocols.src.molecule.Molecule.from_file(file=fnPDB)

        # Add selected modes
        fnModes = [i.getModeFile() for i in self.inputModes.get()]
        modeSelection = np.array(getListFromRangeString(self.modeList.get()))
        self.initStructure.add_modes(files=fnModes, selection=modeSelection)

        # Set Molecule Force field
        if self.coarseGrained.get():
            # If coarse grained, use default Carbon alpha force field values
            self.initStructure.select_atoms(pattern='CA')
            self.initStructure.set_forcefield()
        else:
            # Else read PSF and PRM files
            fnPSF = self.inputPSF.get()
            fnPRM = self.inputPRM.get()
            self.initStructure.set_forcefield(psf_file=fnPSF, prm_file=fnPRM)

    def createTargetDensity(self):
        # Read Volumes
        fnVolumes = [i.getFileName() for i in self.inputVolumes.get()]
        self.targetDensities = []
        for i in fnVolumes:
            targetDensity = continuousflex.protocols.src.density.Volume.from_file(file=i, sigma=self.gauss_sigma.get(),
                                                                           threshold=self.gauss_threshold.get())

            # Rescale the volumes to the initial structure
            targetDensity.rescale(self.initStructure)
            self.targetDensities.append(targetDensity)

    def createFittingParameters(self):
        self.fittingParams = {
            "lb": self.biaisingFactor.get(),
            "lp": self.energyFactor.get(),
            "max_iter": self.n_step.get(),
            "criterion": False,
        }
        self.fittingVars =[]
        if self.fitLocal.get():
            self.fittingVars.append("x")
            self.fittingParams["x_dt"] = self.dtLocal.get()
            self.fittingParams["x_init"] = np.zeros(self.initStructure.coords.shape)
            self.fittingParams["x_mass"] = 1
        if self.fitGlobal.get():
            self.fittingVars.append("q")
            self.fittingParams["q_dt"] = self.dtGlobal.get()
            self.fittingParams["q_init"] = np.zeros(self.initStructure.modes.shape[1])
            self.fittingParams["q_mass"] = 1
        if self.fitRotation.get():
            self.fittingVars.append("angles")
            self.fittingParams["angles_dt"] = self.dtRotation.get()
            self.fittingParams["angles_init"] = np.zeros(3)
            self.fittingParams["angles_mass"] = 1
        if self.fitShift.get():
            self.fittingVars.append("shift")
            self.fittingParams["shift_dt"] = self.dtShift.get()
            self.fittingParams["shift_init"] = np.zeros(3)
            self.fittingParams["shift_mass"] = 1

    def runHMC(self):
        # Run parallel HMC fitting
        self.fittingHMC = continuousflex.protocols.src.functions.multiple_fitting(init=self.initStructure,
                    targets=self.targetDensities, vars=self.fittingVars, n_chain=self.n_chain.get(),
                    n_iter=self.n_iter.get(), n_warmup=self.n_warmup.get(), params=self.fittingParams,
                    n_proc=self.n_proc.get(), verbose=self.verboseLevel.get())

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