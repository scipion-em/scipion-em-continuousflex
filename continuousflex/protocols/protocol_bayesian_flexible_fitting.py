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
from pwem.objects.data import AtomStruct, SetOfVolumes, Volume
# from pyworkflow.utils import replaceExt
# import pwem.emlib.metadata as md

import sys
import os
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

BIASINGENERGY_CC=0
BIASINGENERGY_LS=1

class FlexProtBayesianFlexibleFitting(ProtAnalysis3D):
    """ Protocol for Bayesian Flexible Fitting. """
    _label = 'bayesian flexible fitting'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ########################### Input ############################################
        form.addSection(label='Input')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB", important=True,
                      help='Select the reference PDB.')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select the target EM density volume or set of volumes.')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of normal mode vectors computed by normal mode analysis.')
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
                      help="Select the maximum number of processor to use to perform the fitting.")
        form.addParam('verboseLevel', params.IntParam, default=0, label='Verbose level',
                      help="Print information in the output log for debugging (0->4) ", expertLevel=params.LEVEL_ADVANCED)

        ########################### Force Field ############################################
        form.addSection(label='Force Field')
        form.addParam('atomicModel', params.EnumParam, label="Atomic model", default=ATOMICMODEL_ALLATOMS,
                      choices=['Carbon Alpha', 'Backbone', 'All atoms'],
                      help="Select the desired atomic model to fit the data. \n\t - If \"Carbon Alpha\" is selected, "+
                           "the forcefield is parametrized by defaults forcefield values. \n\t - If \"Backbone\""+
                           " is selected, all hydrogens atoms will be removed from the fitting\n\t - If \"All atoms\""+
                           " is selected, all atoms including hydrogens will be fitted.")
        form.addParam('inputPSF', params.PointerParam, label="Protein Structure File (PSF)", condition='atomicModel>0',
                      pointerClass='EMFile', help='Structure file (.psf). Can be generated with generatePSF protocol ')
        form.addParam('inputPRM', params.FileParam, label="Parameter File (PRM)", condition='atomicModel>0',
                      help='CHARMM force field parameter file (.prm). Can be founded at '+
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')

        form.addParam("bondsPotential", params.BooleanParam, label="Bonds stretching", default=True,
                      help="Bonded interactions : harmonic potential on bond distance.")
        form.addParam("anglesPotential", params.BooleanParam, label="Angles bending", default=True,
                      help="Bonded interactions : harmonic potential on bond angles.")
        form.addParam("ureyPotential", params.BooleanParam, label="Urey-Bradley potential", default=True,
                      help="Bonded interactions : Urey-Bradley potential", condition='atomicModel>0',)
        form.addParam("dihedralsPotential", params.BooleanParam, label="Proper dihedrals", default=True,
                      help="Bonded interactions : cosine potential on dihedral angles.")
        form.addParam("impropersPotential", params.BooleanParam, label="Improper dihedrals", default=True,
                      condition='atomicModel>0', help="Bonded interactions : harmonic potential on improper dihedral angles.")
        form.addParam("vdwPotential", params.BooleanParam, label="Van der Waals", default=False,
                      condition='atomicModel>0', help="Non-bonded interactions : Lennard-Jones potential.")
        form.addParam("elecPotential", params.BooleanParam, label="Electrostatics", default=False,
                      condition='atomicModel>0', help="Non-bonded interactions : Electrostatic potential.")
        form.addParam('cutoffnb', params.FloatParam, default=10.0, label='Non-bonded cutoff (A)',
                      expertLevel=params.LEVEL_ADVANCED, condition="vdwPotential==True or elecPotential==True",
                      help="Cutoff for non-bonded interactions")
        form.addParam('cutoffpl', params.FloatParam, default=15.0, label='Pairlist cutoff (A)',
                      expertLevel=params.LEVEL_ADVANCED, condition="vdwPotential==True or elecPotential==True",
                      help="Cutoff for pairlist generation for the non-bonded interactions. Must be higher than Non-bonded cutoff.")

        ########################### Density ############################################
        form.addSection(label='Density')
        form.addParam('biasedEnergyFunction', params.EnumParam, label='Biasing Energy Function',
                      default=BIASINGENERGY_CC,  choices=['Cross Correlation', 'Least Squares'],
                      help= "Cross Correlation = 1 - CC \n Least Squares = ||sim - exp||^2")
        form.addParam('biaisingFactor', params.FloatParam, default=10000.0, label='Energy constant',
                      help= "Constant factor of the biasing energy : Defines the balance between potential energy and biasing energy"+
                      " \n\t Energy = Energy_potential + k * Energy_biased \n\t For CC, it should be between 1000 and 1000000, "+
                            "For LS, between 0.0001 and 1 (depending on the volume size)")
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="Select the voxel size in Angstrom of the input volumes")
        form.addParam('gauss_sigma', params.FloatParam, default=2.0, label='3D Gaussians standard deviation',
                      help="Standard deviation of the 3D gaussians used for the fitting. ")
        form.addParam('gauss_cutoff', params.FloatParam, default=5, label='3D Gaussians cutoff (A)',
                      help="Cutoff distance in Angstrom from each atoms where kernel are integrated.")


        ########################### Bayesian model ############################################
        form.addSection(label='Bayesian Model')
        form.addParam('fitLocal', params.BooleanParam, label="Fit local dynamics",
                      default=True,
                      help="If selected, the local dynamics will be fitted. Correspond to atomic coordinates displacement.")
        form.addParam('dtLocal', params.FloatParam, default=1,
                      label='Local dynamics time step (fs)',
                      condition='fitLocal==True', expertLevel=params.LEVEL_ADVANCED,
                      help="Integration time step of local dynamics in femto seconds. Higher values will speed up the fitting but can leads to unstability.")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      condition='fitLocal==True', expertLevel=params.LEVEL_ADVANCED,
                      help="Desired instant temperature during the fitting.")
        form.addParam('fitGlobal', params.BooleanParam, label="Fit global dynamics",
                      default=True,
                      help="If selected, the global dynamics will be fitted. Correspond to normal modes displacement.")
        form.addParam('dtGlobal', params.FloatParam, default=0.05,
                      label='Global dynamics time step',
                      condition='fitGlobal==True', expertLevel=params.LEVEL_ADVANCED,
                      help="Integration time step of global dynamics. Higher values will speed up the fitting but can leads to unstability.")
        form.addParam('fitRotation', params.BooleanParam, label="Fit rotations",
                      default=False,
                      help="If selected, the rotations will be fitted. Correspond to rotational displacement.")
        form.addParam('dtRotation', params.FloatParam, default=0.00005,
                      label='Rotations time step',
                      condition='fitRotation==True', expertLevel=params.LEVEL_ADVANCED,
                      help="Integration time step of rotation. Higher values will speed up the fitting but can leads to unstability.")
        form.addParam('fitShift', params.BooleanParam, label="Fit translations",
                      default=False,
                      help="If selected, the shift will be fitted. Correspond to shift displacement.")
        form.addParam('dtShift', params.FloatParam, default=0.00005,
                      label='Translations time step',
                      condition='fitShift==True', expertLevel=params.LEVEL_ADVANCED,
                      help="Integration time step of shift. Higher values will speed up the fitting but can leads to unstability.")

        ########################### HMC Parameters ############################################
        form.addSection(label='Fitting Parameters')
        form.addParam('n_iter', params.IntParam, default=50,  label='Number of iterations',
                      help="Number of HMC loop iterations. ")
        form.addParam('n_warmup', params.IntParam, default=25, label='Number of warmup ',
                      help="Number of HMC warmup loop iterations. The warmup iterations correspond to initial iterations"+
                           " before convergence of the algorithm and will be thorwn away and not taken into account in final results.")
        form.addParam('n_chain', params.IntParam, default=4, label='Number of chains',
                      help="Number of identical parallel HMC run perfomed for each fitting.")
        form.addParam('n_step', params.IntParam, default=20, label='Steps per iteration',
                      help="Number of Molecular Dynamics steps performed for each HMC iterations.")

        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("createInitialStructure")
        self._insertFunctionStep("getInputVolumes")
        self._insertFunctionStep("createTargetDensity")
        self._insertFunctionStep("createFittingParameters")
        self._insertFunctionStep("runHMC")
        self._insertFunctionStep("createOutputStep")

    def createInitialStructure(self):
        # Read PDB
        fnPDB = self.inputPDB.get().getFileName()
        self.initStructure = src.molecule.Molecule(pdb_file=fnPDB)
        self.initStructure.center()

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
            fnPSF = self.inputPSF.get().getFileName()
            fnPRM = self.inputPRM.get()
            self.initStructure.set_forcefield(psf_file=fnPSF, prm_file=fnPRM)
            if self.atomicModel.get() == ATOMICMODEL_BACKBONE:
                self.initStructure.allatoms2backbone()

    def createTargetDensity(self):
        # Read Volumes
        self.targetDensities = []
        for i in self.fnVolumes:
            targetDensity = src.density.Volume.from_file(file=i, sigma=self.gauss_sigma.get(),
                              cutoff=self.gauss_cutoff.get(), voxel_size=self.voxel_size.get())

            # Rescale the volumes to the initial structure
            initDensity = src.density.Volume.from_coords(coord=self.initStructure.coords, size=targetDensity.size,
                                                         sigma = targetDensity.sigma, cutoff=targetDensity.cutoff,
                                                         voxel_size=targetDensity.voxel_size)
            targetDensity.rescale(initDensity)
            self.targetDensities.append(targetDensity)

    def createFittingParameters(self):

        self.fittingParams = {
            "biasing_factor": self.biaisingFactor.get(),
            "gradient" : "LS" if self.biasedEnergyFunction.get() == BIASINGENERGY_LS else "CC",
            "n_step": self.n_step.get(),
            "n_iter" : self.n_iter.get(),
            "n_warmup" : self.n_warmup.get(),
            "criterion": False,
            "temperature" :  self.temperature.get(),
            "potentials": []
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

        if self.bondsPotential.get():
            self.fittingParams["potentials"].append("bonds")
        if self.anglesPotential.get():
            self.fittingParams["potentials"].append("angles")
        if self.ureyPotential.get():
            self.fittingParams["potentials"].append("urey")
        if self.dihedralsPotential.get():
            self.fittingParams["potentials"].append("dihedrals")
        if self.impropersPotential.get():
            self.fittingParams["potentials"].append("impropers")
        if self.vdwPotential.get():
            self.fittingParams["potentials"].append("vdw")
        if self.elecPotential.get():
            self.fittingParams["potentials"].append("elec")
        if self.elecPotential.get() or self.vdwPotential.get():
            self.fittingParams["cutoffnb"] = self.cutoffnb.get()
            self.fittingParams["cutoffpl"] = self.cutoffpl.get()
        if len(self.fittingParams["potentials"]) ==0:
            raise RuntimeError("At least one potential must be used for fitting.")

    def runHMC(self):
        # Run parallel HMC fitting
        # chimera_fit_viewer(mol = self.initStructure,target=self.targetDensities[0])

        models=[]
        for i in range(self.n_volumes):
            models.append(FlexibleFitting(init=self.initStructure, target=self.targetDensities[i], vars=self.fittingVars,n_chain=self.n_chain.get(),
                        params=self.fittingParams, verbose=self.verboseLevel.get(),
                        prefix=self._getExtraPath("fit%i"%i)))

        self.fittingHMC = multiple_fitting(models = models,n_chain=self.n_chain.get(), n_proc=self.n_proc.get())

    def getInputVolumes(self):

        # Get the Volume(s) file name
        if isinstance(self.inputVolumes.get(), Volume):
            self.n_volumes = 1
            self.fnVolumes = [self.inputVolumes.get().getFileName()]
        else:
            self.fnVolumes =[]
            for i in self.inputVolumes.get():
                self.fnVolumes.append(i.getFileName())
        self.n_volumes = len(self.fnVolumes)

        # Convert to MRC if necessary
        for i in range(self.n_volumes):
            pre, ext = os.path.splitext(os.path.basename(self.fnVolumes[i]))
            if ext != ".mrc":
                fnMRC = self._getExtraPath(pre + ".mrc")
                self.fnVolumes[i] = fnMRC
                args = "-i " + self.fnVolumes[i]
                args += " --oext mrc"
                args += " -o " + fnMRC
                self.runJob("xmipp_image_convert",args)

    def createOutputStep(self):
        PDBs = self._createSetOfPDBs("fittedPBDs")
        for i in range(len(self.fittingHMC)) :
            fnPDB = self._getExtraPath(str(i).zfill(5)+'_fitted.pdb')
            self.fittingHMC[i].res["mol"].save_pdb(file=fnPDB)
            PDBs.append(AtomStruct(fnPDB))
        self._defineOutputs(outputSetOfPDBs=PDBs)


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