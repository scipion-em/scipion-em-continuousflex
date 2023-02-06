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

import os.path
import subprocess
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pwem.objects.data import AtomStruct, SetOfAtomStructs, SetOfPDBs, SetOfVolumes,SetOfParticles, Volume
import mrcfile
from pwem.utils import runProgram
from pyworkflow.utils import getListFromRangeString
from .utilities.genesis_utilities import *
from .utilities.pdb_handler import ContinuousFlexPDBHandler
import pyworkflow.utils as pwutils
from pyworkflow.utils import runCommand, buildRunCommand
from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes
from pwem.convert.atom_struct import cifToPdb
from continuousflex import Plugin
from pyworkflow.utils.path import makePath

import pwem.emlib.metadata as md
import re

class FlexProtGenesis(EMProtocol):
    """ Protocol to perform MD/NMMD simulation based on GENESIS. """
    _label = 'MD-NMMD-Genesis'

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)
        self._inputEMMetadata =None

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        # Inputs ============================================================================================
        form.addSection(label='Inputs')

        form.addParam('inputType', params.EnumParam, label="Simulation inputs", default=INPUT_NEW_SIM,
                      choices=['New simulation from topology protocol', 'Restart previous GENESIS simulation', "New simulation from files"],
                      help="Chose the type of input for your simulation",
                      important=True)

        # INPUT_TOPOLOGY
        form.addParam('topoProt', params.PointerParam, label="Input topology protocol",
                      pointerClass="ProtGenerateTopology",
                       help='Provide a generate topology protocol to initialize the simulation',
                      condition="inputType==%i"%INPUT_TOPOLOGY)

        # INPUT_RESTART
        form.addParam('restartProt', params.PointerParam, label="Input GENESIS protocol",
                      pointerClass="FlexProtGenesis",
                       help='Provide a MD-NMMD-GENESIS protocol to restart.', condition="inputType==%i"%INPUT_RESTART)

        # INPUT_NEW_SIM
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB",
                      help='Select the input PDB.', important=True,  condition="inputType==%i"%INPUT_NEW_SIM)
        group = form.addGroup('Forcefield Inputs',  condition="inputType==%i"%INPUT_NEW_SIM)
        group.addParam('forcefield', params.EnumParam, label="Forcefield type", default=FORCEFIELD_CHARMM, important=True,

                      choices=['CHARMM', 'AAGO', 'CAGO'], help="Type of the force field used for energy and force calculation")

        group.addParam('inputTOP', params.FileParam, label="GROMACS Topology File (top)",
                      condition="(forcefield==%i or forcefield==%i)"%(FORCEFIELD_CAGO, FORCEFIELD_AAGO),
                      help='Gromacs ‘top’ file containing information of the system such as atomic masses, charges,'
                           ' atom connectivities. To generate this file for your system, you can either use the protocol'
                           '\" generate topology files\" (SMOG 2 installation is required, https://smog-server.org/smog2/ ),'
                           ' or using SMOG sever (https://smog-server.org/cgi-bin/GenTopGro.pl )')
        group.addParam('inputPRM', params.FileParam, label="CHARMM parameter file (prm)",
                      condition = "forcefield==%i"%FORCEFIELD_CHARMM,
                      help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ' )
        group.addParam('inputRTF', params.FileParam, label="CHARMM topology file (rtf)",
                      condition="forcefield==%i"%FORCEFIELD_CHARMM,
                      help='CHARMM topology file containing information about atom connectivity of residues and'
                           ' other molecules. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')
        group.addParam('inputPSF', params.FileParam, label="CHARMM Structure File (psf)",
                      condition="forcefield==%i"%FORCEFIELD_CHARMM,
                      help='CHARMM/X-PLOR psf file containing information of the system such as atomic masses,'
                            ' charges, and atom connectivities. To generate this file for your system, you can either use the protocol'
                           '\" generate topology files\", VMD psfgen, or online CHARMM GUI ( https://www.charmm-gui.org/ ).')
        group.addParam('inputSTR', params.FileParam, label="CHARMM stream file (str, optional)",
                      condition="forcefield==%i"%FORCEFIELD_CHARMM, default="",
                      help='CHARMM stream file containing both topology information and parameters. '
                           'Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')

        form.addParam('centerPDB', params.BooleanParam, label="Center PDB ?",
                      default=False, help="Center the input PDBs with the center of mass")

        group = form.addGroup('Execution Parameters',expertLevel=params.LEVEL_ADVANCED)
        group.addParam('disableParallelSim', params.BooleanParam, label="Disable parallelisation over the EM data ?", default=False,
                      help="Disabel parallel processing of simualtions over EM input data. Instead, each simulation is run linearly"
                           "with internal parallelization with the specified number of MPI. Running parallel simulation is activated by default "
                           " when using multiple EM data. Running parallel simulation is not available for REUS."
                           "",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('raiseError', params.BooleanParam, label="Stop execution if fails ?", default=True,
                      help="Stop execution if GENESIS program fails",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('md_program', params.EnumParam, label="MD program", default=PROGRAM_ATDYN,
                      choices=['ATDYN', 'SPDYN'],
                      help="SPDYN (Spatial decomposition dynamics) and ATDYN (Atomic decomposition dynamics)"
                    " share almost the same data structures, subroutines, and modules, but differ in"
                    " their parallelization schemes. In SPDYN, the spatial decomposition scheme is implemented with new"
                    " parallel algorithms and GPGPU calculation. In ATDYN, the atomic decomposition scheme"
                    " is introduced for simplicity. The performance of ATDYN is not comparable to SPDYN due to the"
                    " simple parallelization scheme but contains new methods and features. NMMD is available only for ATDYN.",
                      expertLevel=params.LEVEL_ADVANCED)

        # Simulation =================================================================================================
        form.addSection(label='Simulation')
        form.addParam('simulationType', params.EnumParam, label="Simulation type", default=0,
                      choices=['Minimization', 'Molecular Dynamics (MD)', 'Normal Mode Molecular Dynamics (NMMD)', 'Replica-Exchange MD', 'Replica-Exchange NMMD'],
                      help="Type of simulation to be performed by GENESIS", important=True)

        group = form.addGroup('Simulation parameters')
        group.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['Velocity Verlet', 'Leapfrog', ''],
                      help="Type of integrator for the simulation", condition="simulationType!=0")
        group.addParam('time_step', params.FloatParam, default=0.002, label='Time step (ps)',
                      help="Time step in the MD run", condition="simulationType!=0")
        group.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Total number of steps in one MD run")
        group.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="Output frequency for the energy data")
        group.addParam('crdout_period', params.IntParam, default=100, label='Coordinate output period',
                      help="Output frequency for the coordinates data")
        group.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="Update frequency of the non-bonded pairlist",
                      expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('NMMD parameters', condition="simulationType==%i or simulationType==%i"%(SIMULATION_NMMD,
                               SIMULATION_RENMMD))
        group.addParam('inputModes', params.PointerParam, pointerClass="SetOfNormalModes", label='Input Modes',
                       default=None, allowsNull = True,
                       help="Input set of normal modes")
        group.addParam('modeList', params.NumericRangeParam, expertLevel=params.LEVEL_ADVANCED,
                      label="Modes selection", allowsNull = True, default="",
                      help='Select the normal modes that will be used for analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for simulation.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')

        group.addParam('nm_dt', params.FloatParam, label='NM time step', default=0.001,
                      help="Time step of normal modes integration. Should be equal to MD time step. Could be increase "
                           "to accelerate NM integration, however can make the simulation unstable.",
                       condition="simulationType==2 or simulationType==4",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('nm_mass', params.FloatParam, default=10.0, label='NM mass',
                      help="Mass value of Normal modes for NMMD. Lower values accelerate the fitting but can make the "
                           "simulation unstable",   expertLevel=params.LEVEL_ADVANCED)
        group = form.addGroup('REMD parameters', condition="simulationType==%i or simulationType==%i"%(SIMULATION_REMD,
                               SIMULATION_RENMMD))
        group.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="Number of MD steps between replica exchanges",
                       condition="simulationType==%i or simulationType==%i"%(SIMULATION_REMD,
                               SIMULATION_RENMMD))
        group.addParam('nreplica', params.IntParam, default=1, label='Number of replicas',
                      help="Number of replicas for REMD", condition="simulationType==%i or simulationType==%i"%(SIMULATION_REMD,
                               SIMULATION_RENMMD))

        # MD params =================================================================================================
        form.addSection(label='MD parameters')
        group = form.addGroup('Energy')
        group.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['GBSA', 'NONE'],
                      help="Turn on Generalized Born/Solvent accessible surface area model (Implicit Solvent). Boundary condition must be NO."
                           " ATDYN only.")

        group.addParam('boundary', params.EnumParam, label="Boundary", default=0,
                      choices=['No boundary', 'Periodic Boundary Condition'],
                      help="Type of boundary condition. In case of implicit solvent, "
                           " GO models or vaccum simulation, choose No boundary")
        group.addParam('box_size_x', params.FloatParam, label='Box size X',
                      help="Box size along the x dimension", condition="boundary==%i"%BOUNDARY_PBC)
        group.addParam('box_size_y', params.FloatParam, label='Box size Y',
                      help="Box size along the y dimension", condition="boundary==%i"%BOUNDARY_PBC)
        group.addParam('box_size_z', params.FloatParam, label='Box size Z',
                      help="Box size along the z dimension", condition="boundary==%i"%BOUNDARY_PBC)

        group.addParam('electrostatics', params.EnumParam, label="Non-bonded interactions", default=1,
                      choices=['PME', 'Cutoff'],
                      help="Type of Non-bonded interactions. "
                           " CUTOFF: Non-bonded interactions including the van der Waals interaction are just"
                           " truncated at cutoffdist; "
                           " PME : Particle mesh Ewald (PME) method is employed for long-range interactions."
                            " This option is only availabe in the periodic boundary condition")
        group.addParam('vdw_force_switch', params.BooleanParam, label="Switch function Van der Waals", default=True,
                      help="This paramter determines whether the force switch function for van der Waals interactions is"
                        " employed or not. The users must take care about this parameter, when the CHARMM"
                        " force field is used. Typically, vdw_force_switch=YES should be specified in the case of"
                        " CHARMM36",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance',
                      help="Switch-on distance for nonbonded interaction energy/force quenching")
        group.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance',
                      help="Cut-off distance for the non-bonded interactions. This distance must be larger than"
                            " switchdist, while smaller than pairlistdist")
        group.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance',
                      help="Distance used to make a Verlet pair list for non-bonded interactions . This distance"
                            " must be larger than cutoffdist")

        group = form.addGroup('Ensemble', condition="simulationType!=0")
        group.addParam('ensemble', params.EnumParam, label="Ensemble", default=0,
                      choices=['NVT', 'NVE', 'NPT'],
                      help="Type of ensemble, NVE: Microcanonical ensemble, NVT: Canonical ensemble,"
                           " NPT: Isothermal-isobaric ensemble")
        group.addParam('tpcontrol', params.EnumParam, label="Thermostat/Barostat", default=1,
                      choices=['NO', 'LANGEVIN', 'BERENDSEN', 'BUSSI'],
                      help="Type of thermostat and barostat. The availabe algorithm depends on the integrator :"
                           " Leapfrog : BERENDSEN, LANGEVIN;  Velocity Verlet : BERENDSEN (NVT only), LANGEVIN, BUSSI; "
                           " NMMD : LANGEVIN (NVT only)")
        group.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="Initial and target temperature")
        group.addParam('pressure', params.FloatParam, default=1.0, label='Pressure (atm)',
                      help="Target pressure in the NPT ensemble", condition="ensemble==%i"%ENSEMBLE_NPT)

        group = form.addGroup('Contraints', condition="simulationType==%i or simulationType==%i"%(SIMULATION_MD,SIMULATION_REMD))
        group.addParam('rigid_bond', params.BooleanParam, label="Rigid bonds (SHAKE/RATTLE)",
                      default=False,
                      help="Turn on or off the SHAKE/RATTLE algorithms for covalent bonds involving hydrogen. "
                           "Must be False for NMMD.")
        group.addParam('fast_water', params.BooleanParam, label="Fast water (SETTLE)",
                      default=False,
                      help="Turn on or off the SETTLE algorithm for the constraints of the water molecules")
        group.addParam('water_model', params.StringParam, label='Water model', default="TIP3",
                      help="Residue name of the water molecule to be rigidified in the SETTLE algorithm", condition="fast_water")

        # Experiments =================================================================================================
        form.addSection(label='EM data')
        # form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=0,
        #               choices=['None', 'Volume'], important=True,
        #               help="Type of cryo-EM data to be processed")
        form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=0,
                      choices=['None', 'Volume (s)', 'Image (s)'], important=True,
                      help="Type of cryo-EM data to be processed")

        group = form.addGroup('Fitting parameters', condition="EMfitChoice!=0")
        group.addParam('constantK', params.StringParam, default="10000", label='Force constant (kcal/mol)',
                      help="Force constant in Eem = k*(1 - c.c.). Note that in the case of REUS, the number of "
                           " force constant value must be equal to the number of replicas, for example for 4 replicas,"
                           " a valid force constant is \"1000 2000 3000 4000\", otherwise you can specify a range of "
                           " values (for example \"1000-4000\") and the force constant values will be linearly distributed "
                           " to each replica."
                      , condition="EMfitChoice!=0")
        group.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EM Fit Sigma",
                      help="Resolution parameter of the simulated map. This is usually set to the half of the resolution"
                        " of the target map. For example, if the target map resolution is 5 Å, emfit_sigma=2.5",
                      condition="EMfitChoice!=0",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EM Fit Tolerance',
                      help="This variable determines the tail length of the Gaussian function. For example, if em-"
                        " fit_tolerance=0.001 is specified, the Gaussian function is truncated to zero when it is less"
                        " than 0.1% of the maximum value. Smaller value requires large computational cost",
                      condition="EMfitChoice!=0",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('emfit_period', params.IntParam, default=10, label='EM Fit period',
                       help="Number of MD iteration every which the EM poential is updated",
                       condition="EMfitChoice!=0", expertLevel=params.LEVEL_ADVANCED)

        # Volumes
        group = form.addGroup('Volume Parameters', condition="EMfitChoice==%i"%EMFIT_VOLUMES)
        group.addParam('inputVolume', params.PointerParam, pointerClass="Volume",
                      label="Input volume", help='Select the target EM density volume',
                      condition="EMfitChoice==%i"%EMFIT_VOLUMES, important=True)
        group.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="Voxel size in ANgstrom of the target volume", condition="EMfitChoice==%i"%EMFIT_VOLUMES)
        group.addParam('centerOrigin', params.BooleanParam, label="Center Origin", default=True,
                      help="Center the volume to the origin", condition="EMfitChoice==%i"%EMFIT_VOLUMES)
        group.addParam('origin_x', params.FloatParam, default=0, label="Origin X",
                      help="Origin of the first voxel in X direction (in Angstrom) ",
                      condition="EMfitChoice==%i and not centerOrigin"%EMFIT_VOLUMES)
        group.addParam('origin_y', params.FloatParam, default=0, label="Origin Y",
                      help="Origin of the first voxel in Y direction (in Angstrom) ",
                      condition="EMfitChoice==%i and not centerOrigin"%EMFIT_VOLUMES)
        group.addParam('origin_z', params.FloatParam, default=0, label="Origin Z",
                      help="Origin of the first voxel in Z direction (in Angstrom) ",
                      condition="EMfitChoice==%i and not centerOrigin"%EMFIT_VOLUMES)

        # Images
        group = form.addGroup('Image Parameters', condition="EMfitChoice==%i"%EMFIT_IMAGES)
        group.addParam('inputImage', params.PointerParam, pointerClass="SetOfParticles",
                      label="Input images ", help='Select the target image set',
                      condition="EMfitChoice==%i"%EMFIT_IMAGES, important=True)
        group.addParam('pixel_size', params.FloatParam, default=1.0, label='Pixel size (A)',
                      help="Pixel size of the EM data in Angstrom", condition="EMfitChoice==%i"%EMFIT_IMAGES)
        group.addParam('projectAngleChoice', params.EnumParam, default=0, label='Projection angles',
                       choices=['same as image set', 'from xmipp file', 'from other set'],
                      help="Source of projection angles to align the input PDB with the set of images",
                       condition="EMfitChoice==%i"%EMFIT_IMAGES)
        group.addParam('projectAngleXmipp', params.FileParam, default=None, label='projection angle Xmipp file',
                      help="Xmipp metadata file with projection alignement parameters ",
                       condition="EMfitChoice==%i and projectAngleChoice==%i"%(EMFIT_IMAGES,PROJECTION_ANGLE_XMIPP))
        group.addParam('projectAngleImage', params.PointerParam, pointerClass="SetOfParticles",
                      label="projection angle image set  ", help='Image set containing projection alignement parameters',
                      condition="EMfitChoice==%i and projectAngleChoice==%i"%(EMFIT_IMAGES,PROJECTION_ANGLE_IMAGE))

        form.addParallelSection(threads=1, mpi=NUMBER_OF_CPU)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # make path
        self._insertFunctionStep("makePathStep")

        # Create INP files
        self._insertFunctionStep("createGenesisInputStep")

        # Convert input PDB
        self._insertFunctionStep("convertInputPDBStep")

        # Convert normal modes
        if (self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD):
            self._insertFunctionStep("convertNormalModeFileStep")

        # Convert input EM data
        if self.EMfitChoice.get() != EMFIT_NONE:
            self._insertFunctionStep("convertInputEMStep")

        # RUN simulation
        if not self.disableParallelSim.get() and  \
            self.getNumberOfSimulation() >1  and  existsCommand("parallel") :
            self._insertFunctionStep("runSimulationParallel")
        else:
            if not self.disableParallelSim.get() and  \
                self.getNumberOfSimulation() >1  and  not existsCommand("parallel"):
                self.warning("Warning : Can not use parallel computation for GENESIS,"
                                    " please install \"GNU parallel\". Running in linear mode.")
            for i in range(self.getNumberOfSimulation()):
                inp_file = self.getGenesisInputFile(i)
                outPref = self.getOutputPrefix(i)
                self._insertFunctionStep("runSimulation", inp_file, outPref)

        # Create output data
        self._insertFunctionStep("createOutputStep")


    def makePathStep(self):
        makePath(self.getGenFiles())
        makePath(self.getEmdFiles())
        makePath(self._getTmpPath())
    # --------------------------- Convert Input PDBs --------------------------------------------

    def convertInputPDBStep(self):
        """
        Convert input PDB step.
        :return None:
        """

        # COPY PDBS -------------------------------------------------------------
        inputPDBfn = self.getInputPDBfn()
        n_pdb = self.getNumberOfInputPDB()
        for i in range(n_pdb):
            ext = os.path.splitext(inputPDBfn[i])[1]
            if ext == ".pdb" or ext == ".ent":
                runCommand("cp %s %s.pdb" % (inputPDBfn[i], self.getInputPDBprefix(i)))
            elif ext == ".cif" or ext == ".mmcif":
                cifToPdb(inputPDBfn[i], self.getInputPDBprefix(i)+".pdb")
            else:
                print("ERROR (toPdb), Unknown file type for file = %s" % inputPDBfn[i])

        # TOPOLOGY FILES -------------------------------------------------
        inputPrefix = self.getInputPDBprefix()
        if self.getForceField() == FORCEFIELD_CHARMM:
            if self.inputType.get() == INPUT_NEW_SIM:
                inputPSF = self.inputPSF.get()
            elif self.inputType.get() == INPUT_RESTART:
                inputPSF = self.restartProt.get().getInputPDBprefix() + ".psf"
            elif self.inputType.get() == INPUT_TOPOLOGY:
                inputPSF = self.topoProt.get()._getExtraPath("output.psf")
            runCommand("cp %s %s.psf" % (inputPSF, inputPrefix))
        elif self.getForceField() == FORCEFIELD_CAGO or self.getForceField() == FORCEFIELD_AAGO :
            if self.inputType.get() == INPUT_NEW_SIM:
                inputTOP = self.inputTOP.get()
            elif self.inputType.get() == INPUT_RESTART:
                inputTOP = self.restartProt.get().getInputPDBprefix() + ".top"
            elif self.inputType.get() == INPUT_TOPOLOGY:
                inputTOP = self.topoProt.get()._getExtraPath("output.top")
            runCommand("cp %s %s.top" % (inputTOP, inputPrefix))

        # Center PDBs -----------------------------------------------------
        if self.centerPDB.get():
            for i in range(self.getNumberOfInputPDB()):
                cmd = "xmipp_pdb_center -i %s.pdb -o %s.pdb" %\
                        (self.getInputPDBprefix(i),self.getInputPDBprefix(i))
                runCommand(cmd)
                print(cmd)

    def convertNormalModeFileStep(self):
        """
        Convert NM data step
        :return None:
        """
        nm_file = self.getInputPDBprefix() + ".nma"
        if self.modeList.empty():
            modeSelection = np.arange(7,self.inputModes.get().getSize()+1)
        else:
            modeSelection = getListFromRangeString(self.modeList.get())
        with open(nm_file, "w") as f:
            for i in range(self.inputModes.get().getSize()):
                if i+1 in modeSelection:
                    f.write(" VECTOR    %i       VALUE  0.0\n" % (i + 1))
                    f.write(" -----------------------------------\n")
                    nm_vec = np.loadtxt(self.inputModes.get()[i + 1].getModeFile())
                    for j in range(nm_vec.shape[0]):
                        f.write(" %e   %e   %e\n" % (nm_vec[j, 0], nm_vec[j, 1], nm_vec[j, 2]))

    # --------------------------- Convert Input EM data --------------------------------------------

    def convertInputEMStep(self):
        """
        Convert EM data step
        :return None:
        """
        # Convert EM data

        n_em = self.getNumberOfInputEM()
        dest_ext = "mrc" if self.EMfitChoice.get() == EMFIT_VOLUMES else "spi"
        self.readInputEMMetadata()
        runProgram("xmipp_image_convert", "-i %s/inputEM.xmd --oext %s --oroot %s/inputEM_" %
                       (self.getEmdFiles(), dest_ext, self.getEmdFiles()))

        # Fix volumes origin
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            for i in range(n_em):
                # Update mrc header
                volPrefix  = self.getInputEMprefix(i)
                with mrcfile.open("%s.mrc" % volPrefix) as old_mrc:
                    with mrcfile.new("%s.mrc" % volPrefix, overwrite=True) as new_mrc:
                        new_mrc.set_data(old_mrc.data)
                        new_mrc.voxel_size = self.voxel_size.get()
                        new_mrc.header['origin'] = old_mrc.header['origin']
                        if self.centerOrigin.get():
                            origin = -np.array(old_mrc.data.shape) / 2 * self.voxel_size.get()
                            new_mrc.header['origin']['x'] = origin[0]
                            new_mrc.header['origin']['y'] = origin[1]
                            new_mrc.header['origin']['z'] = origin[2]
                        else:
                            new_mrc.header['origin']['x'] = self.origin_x.get()
                            new_mrc.header['origin']['y'] = self.origin_y.get()
                            new_mrc.header['origin']['z'] = self.origin_z.get()
                        new_mrc.update_header_from_data()
                        new_mrc.update_header_stats()

    # --------------------------- GENESIS step --------------------------------------------

    def createGenesisInputStep(self):
        """
        Create GENESIS input files
        :return None:
        """
        for indexFit in range(self.getNumberOfSimulation()):
            # INP file name
            inp_file = self.getGenesisInputFile(indexFit)
            args = self.getDefaultArgs(indexFit)
            createGenesisInput(inp_file, **args)

    def getDefaultArgs(self, indexFit=0):
        """
        get default argument to run GENESIS
        @param indexFit:
        @return:

        """
        inputRTF, inputPRM, inputSTR = self.getCHARMMInputs()
        args = {
            # Inputs files
            "outputPrefix": self.getOutputPrefix(indexFit),
            "inputPDBprefix": self.getInputPDBprefix(indexFit),
            "inputEMprefix": self.getInputEMprefix(indexFit),
            "rstFile": self.getRestartFile(indexFit),
            "nm_number": self.getNumberOfNormalModes(),
            "rigid_body_params": self.getRigidBodyParams(indexFit),
            "forcefield": self.getForceField(),
            "inputRTF": inputRTF,
            "inputPRM": inputPRM,
            "inputSTR": inputSTR,

            # Input Params
            "inputType": self.inputType.get(),
            "simulationType": self.simulationType.get(),
            "electrostatics": self.electrostatics.get(),
            "switch_dist": self.switch_dist.get(),
            "cutoff_dist": self.cutoff_dist.get(),
            "pairlist_dist": self.pairlist_dist.get(),
            "vdw_force_switch": self.vdw_force_switch.get(),
            "implicitSolvent": self.implicitSolvent.get(),
            "integrator": self.integrator.get(),
            "time_step": self.time_step.get(),
            "eneout_period": self.eneout_period.get(),
            "crdout_period": self.crdout_period.get(),
            "n_steps": self.n_steps.get(),
            "nbupdate_period": self.nbupdate_period.get(),
            "nm_dt": self.nm_dt.get(),
            "nm_mass": self.nm_mass.get(),
            "rigid_bond": self.rigid_bond.get(),
            "fast_water": self.fast_water.get(),
            "water_model": self.water_model.get(),
            "box_size_x": self.box_size_x.get(),
            "box_size_y": self.box_size_y.get(),
            "box_size_z": self.box_size_z.get(),
            "boundary": self.boundary.get(),
            "ensemble": self.ensemble.get(),
            "tpcontrol": self.tpcontrol.get(),
            "temperature": self.temperature.get(),
            "pressure": self.pressure.get(),
            "EMfitChoice": self.EMfitChoice.get(),
            "constantK": self.constantK.get(),
            "nreplica": self.nreplica.get(),
            "emfit_sigma": self.emfit_sigma.get(),
            "emfit_tolerance": self.emfit_tolerance.get(),
            "emfit_period": self.emfit_period.get(),
            "pixel_size": self.pixel_size.get(),
            "exchange_period": self.exchange_period.get()
        }
        return args

    def runSimulation(self, inp_file, outPref):
        """
        Run GENESIS simulations
        :return None:
        """
        programname = "atdyn" if self.md_program.get() == PROGRAM_ATDYN else "spdyn"
        params = "%s > %s.log" % (inp_file,outPref)
        env = self.getGenesisEnv()
        env.set("OMP_NUM_THREADS",str(self.numberOfThreads.get()))
        command = buildRunCommand(programname, params, numberOfMpi=self.numberOfMpi.get(),
                              hostConfig=self._stepsExecutor.hostConfig,
                              env=env)
        command = Plugin.getContinuousFlexCmd(command)
        runCommand(command, env=env)

    def runSimulationParallel(self):
        """
        Run multiple GENESIS simulations in parallel
        :return None:
        """

        # Set number of MPI per fit
        if self.getNumberOfSimulation() <= self.numberOfMpi.get():
            numberOfMpiPerFit   = self.numberOfMpi.get()//self.getNumberOfSimulation()
        else:
            if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
                nreplica = self.nreplica.get()
                if nreplica > self.numberOfMpi.get():
                    raise RuntimeError("Number of MPI cores should be larger than the number of replicas.")
            else:
                nreplica = 1
            numberOfMpiPerFit   = nreplica

        # Set environnement
        env = self.getGenesisEnv()
        env.set("OMP_NUM_THREADS",str(self.numberOfThreads.get()))

        # Build command
        programname = os.path.join( Plugin.getVar("GENESIS_HOME"), "bin/atdyn")
        outPath, outName = os.path.split(self.getOutputPrefix())
        outLog = os.path.join(outPath,re.sub("output_\d+", "output_{}",outName))
        params = "%s/INP_{} > %s.log " %(self.getGenFiles(), outLog)
        cmd = buildRunCommand(programname, params, numberOfMpi=numberOfMpiPerFit, hostConfig=self._stepsExecutor.hostConfig,
                              env=env)
        # Build parallel command
        parallel_cmd = "seq -f \"%%06g\" 1 %i | parallel -P %i \" %s\" " % (
        self.getNumberOfSimulation(),self.numberOfMpi.get()//numberOfMpiPerFit, cmd)
        parallel_cmd = Plugin.getContinuousFlexCmd(parallel_cmd)

        print("Command : %s" % cmd)
        print("Parallel Command : %s" % parallel_cmd)
        try :
            runCommand(parallel_cmd, env=env)
        except subprocess.CalledProcessError :
            print("Warning : Some processes returned with errors")



    # --------------------------- Create output step --------------------------------------------

    def createOutputStep(self):
        """
        Create output PDB or set of PDBs
        :return None:
        """
        # Extract the pdb from the DCD file in case of SPDYN
        if self.md_program.get() == PROGRAM_SPDYN:
            for i in range(self.getNumberOfSimulation()):
                outputPrefix = self.getOutputPrefixAll(i)
                for j in outputPrefix:
                        lastPDBFromDCD(
                            inputDCD=j + ".dcd",
                            outputPDB=j + ".pdb",
                            inputPDB=self.getInputPDBprefix(i) + ".pdb")

        # In Case of CAGO, replace PDB info by input PDB because Genesis is not saving it properly
        if self.getForceField() == FORCEFIELD_CAGO:
            input = ContinuousFlexPDBHandler(self.getInputPDBprefix() + ".pdb")
            for i in range(self.getNumberOfSimulation()):
                outputPrefix = self.getOutputPrefixAll(i)
                for j in outputPrefix:
                    fn_output = j + ".pdb"
                    if os.path.exists(fn_output) and os.path.getsize(fn_output) !=0:
                        output = ContinuousFlexPDBHandler(fn_output)
                        input.coords = output.coords
                        input.write_pdb(j + ".pdb")

        # CREATE a output PDB
        if (self.simulationType.get() != SIMULATION_REMD  and self.simulationType.get() != SIMULATION_RENMMD )\
                and self.getNumberOfSimulation() == 1:
            self._defineOutputs(outputPDB=AtomStruct(self.getOutputPrefix() + ".pdb"))

        # CREATE SET OF output PDBs
        else:
            missing_pdbs = []
            pdbset = self._createSetOfPDBs("outputPDBs")
            # Add each output PDB to the Set
            for i in range(self.getNumberOfSimulation()):
                outputPrefix =self.getOutputPrefixAll(i)
                for j in outputPrefix:
                    pdb_fname = j + ".pdb"
                    if os.path.isfile(pdb_fname) and os.path.getsize(pdb_fname) != 0 :
                        pdbset.append(AtomStruct(pdb_fname))
                    else:
                        missing_pdbs.append(i+1)
            self._defineOutputs(outputPDBs=pdbset)

            # If some pdbs are missing, output a subset of the EM data that have actually been anaylzed
            if self.EMfitChoice.get() != EMFIT_NONE and len( missing_pdbs)>0:
                if self.EMfitChoice.get() == EMFIT_VOLUMES:
                    inSet = self.inputVolume.get()
                    outSet = self._createSetOfVolumes("subsetVolumes")
                    outSet.setSamplingRate(self.voxel_size.get())

                else:
                    inSet = self.inputImage.get()
                    outSet = self._createSetOfParticles("subsetParticles")
                    outSet.setSamplingRate(self.pixel_size.get())

                for i in inSet:
                    if not i.getObjId() in missing_pdbs:
                        outSet.append(i)
                self._defineOutputs(subsetEM=outSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Genesis in a software for Molecular Dynamics Simulation, "
                   "Normal Mode Molecular Dynamics (NMMD), Replica Exchange Umbrela "
                   "Sampling (REUS) and Energy Minimization"]
        return summary

    def _validate(self):
        errors = []
        print(self.getGenesisEnv()["PATH"])
        if not os.path.exists(os.path.join(
                Plugin.getVar("GENESIS_HOME"), 'bin/atdyn')):
            errors.append("Missing GENESIS program : atdyn ")

        if not os.path.exists(os.path.join(
                Plugin.getVar("GENESIS_HOME"), 'bin/spdyn')):
            errors.append("Missing GENESIS program : spdyn ")

        return errors

    def _citations(self):
        return ["kobayashi2017genesis","vuillemot2022NMMD","harastani2022continuousflex"]

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------

    def getNumberOfInputPDB(self):
        """
        Get the number of input PDBs
        :return int: number of input PDBs
        """
        return len(self.getInputPDBfn())

    def getNumberOfInputEM(self):
        """
        Get the number of input EM data to analyze
        :return int : number of input EM data
        """
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), SetOfVolumes): return self.inputVolume.get().getSize()
            else: return 1
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            if isinstance(self.inputImage.get(), SetOfParticles): return self.inputImage.get().getSize()
            else: return 1
        else: return 0

    def getNumberOfSimulation(self):
        """
        Get the number of simulations to perform
        :return int: Number of simulations
        """
        numberOfInputPDB = self.getNumberOfInputPDB()
        numberOfInputEM = self.getNumberOfInputEM()

        # Check input volumes/images correspond to input PDBs
        if numberOfInputPDB != numberOfInputEM and \
                numberOfInputEM != 1 and numberOfInputPDB != 1 \
                and numberOfInputEM != 0:
            raise RuntimeError("Number of input EM data and PDBs must be the same.")
        return np.max([numberOfInputEM, numberOfInputPDB])

    def getNumberOfNormalModes(self):
        if self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD :
            if self.modeList.empty():
                modeSelection = np.arange(7,self.inputModes.get().getSize()+1)
            else:
                modeSelection\
                    = getListFromRangeString(self.modeList.get())
            return len(modeSelection)
        else:
            return None

    def getInputPDBfn(self):
        """
        Get the input PDB file names
        :return list : list of input PDB file names
        """
        initFn = []

        if self.inputType.get() == INPUT_RESTART:
            for i in range(self.restartProt.get().getNumberOfSimulation()):
                initFn += self.restartProt.get().getOutputPrefixAll(i)
            initFn = [i+".pdb" for i in initFn]
        elif self.inputType.get() == INPUT_TOPOLOGY:
            initFn = [self.topoProt.get().outputPDB.getFileName()]
        else:
            if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                    isinstance(self.inputPDB.get(), SetOfPDBs):
                for i in range(self.inputPDB.get().getSize()):
                    initFn.append(self.inputPDB.get()[i+1].getFileName())

            else:
                initFn.append(self.inputPDB.get().getFileName())
        return initFn

    def getGenesisInputFile(self, index=0):
        return "%s/INP_%s" % (self.getGenFiles(),str(index + 1).zfill(6))

    def getInputPDBprefix(self, index=0):
        """
        Get the input PDB prefix of the specified index
        :param int index: index of input PDB
        :return str: Input PDB prefix
        """
        prefix = self._getExtraPath("inputPDB_%s")
        if self.getNumberOfInputPDB() == 1:
            return prefix % str(1).zfill(6)
        else:
            return prefix % str(index + 1).zfill(6)

    def getInputEMprefix(self, index=0):
        """
        Get the input EM data prefix of the specified index
        :param int index: index of the EM data
        :return str: Input EM data prefix
        """
        prefix = self.getEmdFiles()+"/inputEM_%s"
        if self.getNumberOfInputEM() == 0:
            return ""
        elif self.getNumberOfInputEM() == 1:
            return prefix % str(1).zfill(6)
        else:
            return prefix % str(index + 1).zfill(6)


    def getOutputPrefix(self, index=0):
        """
        Output prefix of the specified index
        :param int index: index of the simulation to get
        :return string : Output prefix of the specified index
        """
        return self._getExtraPath("output_%s" % str(index + 1).zfill(6))

    def getOutputPrefixAll(self, index=0):
        """
        All output prefix of the specified index including multiple replicas in case of REUS
        :param int index: index of the simulation to get
        :return list: list of all output prefix of the specified index
        """
        outputPrefix=[]
        if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
            for i in range(self.nreplica.get()):
                outputPrefix.append(self._getExtraPath("output_%s_remd%i" %
                                (str(index + 1).zfill(6), i + 1)))
        else:
            outputPrefix.append(self._getExtraPath("output_%s" % str(index + 1).zfill(6)))
        return outputPrefix
    def getEmdFiles(self):
        return self._getExtraPath("EM_data")
    def getGenFiles(self):
        return self._getExtraPath("genesis_inputs")

    def getRigidBodyParams(self, index=0):
        """
        Get the current rigid body parameters for the specified index in case of EMFIT with iamges
        :param int index: Index of the simulation
        :return list: angle_rot, angle_tilt, angle_psi, shift_x, shift_y
        """
        if self.EMfitChoice.get() == EMFIT_IMAGES :
            inputMd = self.getInputEMMetadata()

            idx = int(index + 1)
            params =  [
                inputMd.getValue(md.MDL_ANGLE_ROT, idx),
                inputMd.getValue(md.MDL_ANGLE_TILT, idx),
                inputMd.getValue(md.MDL_ANGLE_PSI, idx),
                inputMd.getValue(md.MDL_SHIFT_X, idx),
                inputMd.getValue(md.MDL_SHIFT_Y, idx),
            ]
            if any([i is None for i in params]):
                raise RuntimeError("Can not find angles or shifts")
            return params
        else:
            return None


    def getGenesisEnv(self):
        """
        Get environnement for running GENESIS
        :return Environ: environnement
        """
        environ = pwutils.Environ(os.environ)
        environ.set('PATH', os.path.join(Plugin.getVar("GENESIS_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        environ.update({'LD_LIBRARY_PATH': Plugin.getCondaLibPath()}, position=pwutils.Environ.BEGIN)
        return environ

    def getRestartFile(self, index=0):
        """
        Get input restart file
        :param int index: Index of the simulation
        :return str: restart file
        """
        if self.inputType.get() == INPUT_RESTART:
            if len(self.restartProt.get().getOutputPrefixAll(index))>1:
                raise RuntimeError("Multiple restart not implemented")
            rstfile = self.getInputPDBprefix(index) + ".rst"
            if not os.path.exists(rstfile):
                runCommand("cp %s.rst %s" % (self.restartProt.get().getOutputPrefix(), rstfile))
            return rstfile
        else:
            return None

    def getForceField(self):
        """
        Get simulation forcefield
        :return int: forcefield
        """
        if self.inputType.get() == INPUT_RESTART:
            return self.restartProt.get().getForceField()
        elif self.inputType.get() == INPUT_TOPOLOGY:
            return self.topoProt.get().forcefield.get()
        else:
            return self.forcefield.get()

    def getInputEMMetadata(self):
        if self._inputEMMetadata is None:
            self._inputEMMetadata = self.readInputEMMetadata()
        return self._inputEMMetadata

    def readInputEMMetadata(self):
        nameMd = "%s/inputEM.xmd"%self.getEmdFiles()

        if self.EMfitChoice.get() == EMFIT_IMAGES:
            writeSetOfParticles(self.inputImage.get(), nameMd)
            inputEMMetadata = md.MetaData(nameMd)
            if self.projectAngleChoice.get() == PROJECTION_ANGLE_XMIPP:
                xmd = md.MetaData(self.projectAngleXmipp.get())
                for i in xmd:
                    rot = xmd.getValue(md.MDL_ANGLE_ROT, i)
                    tilt = xmd.getValue(md.MDL_ANGLE_TILT, i)
                    psi = xmd.getValue(md.MDL_ANGLE_PSI, i)
                    shx = xmd.getValue(md.MDL_SHIFT_X, i)
                    shy = xmd.getValue(md.MDL_SHIFT_Y, i)
                    inputEMMetadata.setValue(md.MDL_ANGLE_ROT, rot, i)
                    inputEMMetadata.setValue(md.MDL_ANGLE_TILT, tilt, i)
                    inputEMMetadata.setValue(md.MDL_ANGLE_PSI, psi, i)
                    inputEMMetadata.setValue(md.MDL_SHIFT_X, shx, i)
                    inputEMMetadata.setValue(md.MDL_SHIFT_Y, shy, i)
                inputEMMetadata.write(nameMd)
            elif self.projectAngleChoice.get() == PROJECTION_ANGLE_IMAGE:
                raise RuntimeError("projection angles from other image set error : Not implemented")

        elif self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), Volume):
                inputEMMetadata = md.MetaData()
                inputEMMetadata.setValue(md.MDL_IMAGE,
                                              self.inputVolume.get().getFileName(), inputEMMetadata.addObject())
                inputEMMetadata.write(nameMd)
            else:
                writeSetOfVolumes(self.inputVolume.get(), nameMd)
                inputEMMetadata = md.MetaData(nameMd)
        return inputEMMetadata

    def getCHARMMInputs(self):
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            if self.inputType.get() == INPUT_RESTART:
                return self.restartProt.get().getCHARMMInputs()
            elif  self.inputType.get() == INPUT_TOPOLOGY:
                return self.topoProt.get().inputRTF.get(),self.topoProt.get().inputPRM.get(), self.topoProt.get().inputSTR.get()
            elif  self.inputType.get() == INPUT_NEW_SIM:
                return self.inputRTF.get(),self.inputPRM.get(), self.inputSTR.get()
        else:
            return None,None,None

def createGenesisInput(inp_file, outputPrefix="", inputPDBprefix="", inputEMprefix="", rstFile="", nm_number=0,
                       rigid_body_params=None, forcefield= FORCEFIELD_CAGO, inputRTF=None, inputPRM=None,
                       inputSTR=None, inputType=INPUT_NEW_SIM, simulationType=SIMULATION_MIN,
                       electrostatics=ELECTROSTATICS_CUTOFF, switch_dist=10.0, cutoff_dist=12.0,
                       pairlist_dist=15.0, vdw_force_switch=True, implicitSolvent=IMPLICIT_SOLVENT_NONE,
                       integrator=INTEGRATOR_LEAPFROG, time_step=0.001, eneout_period=100, crdout_period=100,
                       n_steps=10000, nbupdate_period=10, nm_dt=0.001, nm_mass=10.0, rigid_bond=False,
                       fast_water = False, water_model="TIP3", box_size_x=None, box_size_y=None, box_size_z=None,
                       boundary=BOUNDARY_NOBC, ensemble=ENSEMBLE_NVE, tpcontrol=TPCONTROL_NONE, temperature=300.0,
                       pressure=1.0, EMfitChoice=EMFIT_NONE, constantK=1000.0, nreplica=4, emfit_sigma=2.0,
                       emfit_tolerance=0.01, emfit_period=10, pixel_size=1.0, exchange_period=100):
    s = "\n[INPUT] \n"  # -----------------------------------------------------------
    s += "pdbfile = %s.pdb\n" % inputPDBprefix
    if forcefield == FORCEFIELD_CHARMM:
        s += "psffile = %s.psf\n" % inputPDBprefix
        s += "topfile = %s\n" % inputRTF
        s += "parfile = %s\n" % inputPRM
        if inputSTR != "" and inputSTR is not None:
            s += "strfile = %s\n" % inputSTR
    elif forcefield == FORCEFIELD_AAGO or forcefield == FORCEFIELD_CAGO:
        s += "grotopfile = %s.top\n" % inputPDBprefix
    if inputType == INPUT_RESTART:
        s += "rstfile = %s \n" % rstFile

    s += "\n[OUTPUT] \n"  # -----------------------------------------------------------
    if simulationType == SIMULATION_REMD or simulationType == SIMULATION_RENMMD:
        s += "remfile = %s_remd{}.rem\n" % outputPrefix
        s += "logfile = %s_remd{}.log\n" % outputPrefix
        s += "dcdfile = %s_remd{}.dcd\n" % outputPrefix
        s += "rstfile = %s_remd{}.rst\n" % outputPrefix
        s += "pdbfile = %s_remd{}.pdb\n" % outputPrefix
    else:
        s += "dcdfile = %s.dcd\n" % outputPrefix
        s += "rstfile = %s.rst\n" % outputPrefix
        s += "pdbfile = %s.pdb\n" % outputPrefix

    s += "\n[ENERGY] \n"  # -----------------------------------------------------------
    if forcefield == FORCEFIELD_CHARMM:
        s += "forcefield = CHARMM \n"
    elif forcefield == FORCEFIELD_AAGO:
        s += "forcefield = AAGO  \n"
    elif forcefield == FORCEFIELD_CAGO:
        s += "forcefield = CAGO  \n"

    if electrostatics == ELECTROSTATICS_CUTOFF:
        s += "electrostatic = CUTOFF  \n"
    else:
        s += "electrostatic = PME  \n"
    s += "switchdist   = %.2f \n" % switch_dist
    s += "cutoffdist   = %.2f \n" % cutoff_dist
    s += "pairlistdist = %.2f \n" % pairlist_dist
    if vdw_force_switch:
        s += "vdw_force_switch = YES \n"
    if implicitSolvent == IMPLICIT_SOLVENT_GBSA:
        s += "implicit_solvent = GBSA \n"
        s += "gbsa_eps_solvent = 78.5 \n"
        s += "gbsa_eps_solute  = 1.0 \n"
        s += "gbsa_salt_cons   = 0.2 \n"
        s += "gbsa_surf_tens   = 0.005 \n"

    if simulationType == SIMULATION_MIN:
        s += "\n[MINIMIZE]\n"  # -----------------------------------------------------------
        s += "method = SD\n"
    else:
        s += "\n[DYNAMICS] \n"  # -----------------------------------------------------------
        if simulationType == SIMULATION_NMMD or simulationType == SIMULATION_RENMMD:
            s += "integrator = NMMD  \n"
        elif integrator == INTEGRATOR_VVERLET:
            s += "integrator = VVER  \n"
        elif integrator == INTEGRATOR_LEAPFROG:
            s += "integrator = LEAP  \n"

        s += "timestep = %f \n" % time_step
    s += "nsteps = %i \n" % n_steps
    s += "eneout_period = %i \n" % eneout_period
    s += "crdout_period = %i \n" % crdout_period
    s += "rstout_period = %i \n" % n_steps
    s += "nbupdate_period = %i \n" % nbupdate_period

    if simulationType == SIMULATION_NMMD or simulationType == SIMULATION_RENMMD:
        s += "\n[NMMD] \n"  # -----------------------------------------------------------
        s += "nm_number = %i \n" % nm_number
        s += "nm_mass = %f \n" % nm_mass
        s += "nm_file = %s.nma \n" % inputPDBprefix
        if nm_dt is None:
            s += "nm_dt = %f \n" % time_step
        else:
            s += "nm_dt = %f \n" % nm_dt

    if simulationType != SIMULATION_MIN:
        s += "\n[CONSTRAINTS] \n"  # -----------------------------------------------------------
        if rigid_bond:
            s += "rigid_bond = YES \n"
        else:
            s += "rigid_bond = NO \n"
        if fast_water:
            s += "fast_water = YES \n"
            s += "water_model = %s \n" % water_model
        else:
            s += "fast_water = NO \n"

    s += "\n[BOUNDARY] \n"  # -----------------------------------------------------------
    if boundary == BOUNDARY_PBC:
        s += "type = PBC \n"
        s += "box_size_x = %f \n" % box_size_x
        s += "box_size_y = %f \n" % box_size_y
        s += "box_size_z = %f \n" % box_size_z
    else:
        s += "type = NOBC \n"

    if simulationType != SIMULATION_MIN:
        s += "\n[ENSEMBLE] \n"  # -----------------------------------------------------------
        if ensemble == ENSEMBLE_NVE:
            s += "ensemble = NVE  \n"
        elif ensemble == ENSEMBLE_NPT:
            s += "ensemble = NPT  \n"
        else:
            s += "ensemble = NVT  \n"
        if tpcontrol == TPCONTROL_LANGEVIN:
            s += "tpcontrol = LANGEVIN  \n"
        elif tpcontrol == TPCONTROL_BERENDSEN:
            s += "tpcontrol = BERENDSEN  \n"
        elif tpcontrol == TPCONTROL_BUSSI:
            s += "tpcontrol = BUSSI  \n"
        else:
            s += "tpcontrol = NO  \n"
        s += "temperature = %.2f \n" % temperature
        if ensemble == ENSEMBLE_NPT:
            s += "pressure = %.2f \n" % pressure

    if (EMfitChoice == EMFIT_VOLUMES or EMfitChoice == EMFIT_IMAGES) \
            and simulationType != SIMULATION_MIN:
        s += "\n[SELECTION] \n"  # -----------------------------------------------------------
        s += "group1 = all and not hydrogen\n"

        s += "\n[RESTRAINTS] \n"  # -----------------------------------------------------------
        s += "nfunctions = 1 \n"
        s += "function1 = EM \n"
        constStr = constantK
        if "-" in constStr:
            splt = constStr.split("-")
            constStr = " ".join(
                [str(int(i)) for i in np.linspace(int(splt[0]), int(splt[1]), nreplica)])
        s += "constant1 = %s \n" % constStr
        s += "select_index1 = 1 \n"

        s += "\n[EXPERIMENTS] \n"  # -----------------------------------------------------------
        s += "emfit = YES  \n"
        s += "emfit_sigma = %.4f \n" % emfit_sigma
        s += "emfit_tolerance = %.6f \n" % emfit_tolerance
        s += "emfit_period = %i  \n" % emfit_period
        if EMfitChoice == EMFIT_VOLUMES:
            s += "emfit_target = %s.mrc \n" % inputEMprefix
        elif EMfitChoice == EMFIT_IMAGES:
            s += "emfit_type = IMAGE \n"
            s += "emfit_target = %s.spi \n" % inputEMprefix
            s += "emfit_pixel_size =  %f\n" % pixel_size
            s += "emfit_roll_angle = %f\n" % rigid_body_params[0]
            s += "emfit_tilt_angle = %f\n" % rigid_body_params[1]
            s += "emfit_yaw_angle =  %f\n" % rigid_body_params[2]
            s += "emfit_shift_x = %f\n" % rigid_body_params[3]
            s += "emfit_shift_y =  %f\n" % rigid_body_params[4]

        if simulationType == SIMULATION_REMD or simulationType == SIMULATION_RENMMD:
            s += "\n[REMD] \n"  # -----------------------------------------------------------
            s += "dimension = 1 \n"
            s += "exchange_period = %i \n" % exchange_period
            s += "type1 = RESTRAINT \n"
            s += "nreplica1 = %i \n" % nreplica
            s += "rest_function1 = 1 \n"

    with open(inp_file, "w") as f:
        f.write(s)