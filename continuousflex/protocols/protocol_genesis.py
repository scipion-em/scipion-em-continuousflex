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
from pyworkflow.utils.path import createLink

import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pwem.objects.data import AtomStruct, SetOfAtomStructs, SetOfPDBs, SetOfVolumes,SetOfParticles

import numpy as np
import mrcfile
from pwem.emlib.image import ImageHandler
from pwem.utils import runProgram
from pyworkflow.utils import getListFromRangeString
import xmipp3.convert


from .utilities.genesis_utilities import *
from .utilities.pdb_handler import ContinuousFlexPDBHandler

from xmipp3 import Plugin
import pyworkflow.utils as pwutils
from pyworkflow.utils import runCommand, buildRunCommand

class ProtGenesis(EMProtocol):
    """ Protocol to perform MD/NMMD simulation based on GENESIS. """
    _label = 'MD-NMMD-Genesis'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        # Inputs ============================================================================================
        form.addSection(label='Inputs')

        form.addParam('restartChoice', params.BooleanParam, label="Restart GENESIS protocol ?", default=False,
                      help="Restart a previous GENESIS simulation. ")


        form.addParam('restartProt', params.PointerParam, label="Input GENESIS protocol",pointerClass="ProtGenesis",
                       help='Provide a GENESIS protocol to restart.', condition="restartChoice" ,important=True)

        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct,SetOfAtomStructs,SetOfPDBs', label="Input PDB (s)",
                      help='Select the input PDB.', important=True, condition="not restartChoice" )
        form.addParam('centerPDB', params.BooleanParam, label="Center PDB ?",
                      default=False, help="Center the input PDBs with the center of mass", condition="not restartChoice" )

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


        group = form.addGroup('Forcefield Inputs', condition="not restartChoice" )
        group.addParam('forcefield', params.EnumParam, label="Forcefield type", default=0, important=True,
                      choices=['CHARMM', 'AAGO', 'CAGO'], help="Type of the force field used for energy and force calculation")
        group.addParam('generateTop', params.BooleanParam, label="Generate topology files ?",
                      default=False, help="Use the GUI to generate topology files for you (PSF file for CHARMM and TOP file for AAGO/CAGO)."
                                          " Requires VMD psfgen for CHARMM forcefields "
                                          " and SMOG2 for GO models. Note that the generated topology files will not include"
                                          " solvent.")
        group.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=0,
                      choices=['NO', 'RNA', 'DNA'], condition ="generateTop",
                       help="Specify if the generator should consider nucleic residues as DNA or RNA")
        group.addParam('smog_dir', params.FileParam, label="SMOG 2 install directory",
                      help="Path to SMOG2 install directory (For SMOG2 installation, see "
                       "https://smog-server.org/smog2/ , otherwise use the web GUI "
                       "https://smog-server.org/cgi-bin/GenTopGro.pl )", condition="(forcefield==1 or forcefield==2) and generateTop")
        group.addParam('inputTOP', params.FileParam, label="GROMACS Topology File (top)",
                      condition="(forcefield==1 or forcefield==2) and not generateTop",
                      help='Gromacs ‘top’ file containing information of the system such as atomic masses, charges,'
                           ' atom connectivities. To generate this file for your system, you can either use the option'
                           '\" generate topology files\" (SMOG 2 installation is required, https://smog-server.org/smog2/ ),'
                           ' or using SMOG sever (https://smog-server.org/cgi-bin/GenTopGro.pl )')
        group.addParam('inputPRM', params.FileParam, label="CHARMM parameter file (prm)",
                      condition = "forcefield==0",
                      help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ' )
        group.addParam('inputRTF', params.FileParam, label="CHARMM topology file (rtf)",
                      condition="forcefield==0 or ((forcefield==1 or forcefield==2) and generateTop)",
                      help='CHARMM topology file containing information about atom connectivity of residues and'
                           ' other molecules. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml '
                           ' Note: In the case of generating topology files for GO models (SMOG2), '
                           ' the CHARMM topology file and VMD psfgen are used to fill missing atoms/residues.')
        group.addParam('inputPSF', params.FileParam, label="CHARMM Structure File (psf)",
                      condition="forcefield==0 and not generateTop",
                      help='CHARMM/X-PLOR psf file containing information of the system such as atomic masses,'
                            ' charges, and atom connectivities. To generate this file for your system, you can either use the option'
                           '\" generate topology files\", VMD psfgen, or online CHARMM GUI ( https://www.charmm-gui.org/ ).')
        group.addParam('inputSTR', params.FileParam, label="CHARMM stream file (str)",
                      condition="forcefield==0", default="",
                      help='CHARMM stream file containing both topology information and parameters. '
                           'Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ',
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

        group = form.addGroup('NMMD parameters', condition="simulationType==2 or simulationType==4")
        group.addParam('nm_number', params.IntParam, default=10, label='Number of normal modes',
                      help="Number of normal modes for NMMD. 10 should work in most cases. Avoid "
                           " using too much NM (>50).",
                      condition="simulationType==2 or simulationType==4")
        group.addParam('inputModes', params.PointerParam, pointerClass = "SetOfNormalModes", label='Input Modes', default=None,
                      help="Input set of normal modes", condition="simulationType==2 or simulationType==4")
        group.addParam('nm_dt', params.FloatParam, label='NM time step', default=0.001,
                      help="TODO", condition="simulationType==2 or simulationType==4",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('nm_mass', params.FloatParam, default=10.0, label='NM mass',
                      help="Mass value of Normal modes for NMMD", condition="simulationType==2 or simulationType==4",
                      expertLevel=params.LEVEL_ADVANCED)
        group.addParam('nm_init', params.FileParam, label='NM init', default=None,
                      help="TODO", condition="simulationType==2 or simulationType==4",expertLevel=params.LEVEL_ADVANCED)
        group = form.addGroup('REMD parameters', condition="simulationType==3 or simulationType==4")
        group.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="Number of MD steps between replica exchanges", condition="simulationType==3 or simulationType==4")
        group.addParam('nreplica', params.IntParam, default=1, label='Number of replicas',
                      help="Number of replicas for REMD", condition="simulationType==3 or simulationType==4")

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
                      help="Box size along the x dimension", condition="boundary==1")
        group.addParam('box_size_y', params.FloatParam, label='Box size Y',
                      help="Box size along the y dimension", condition="boundary==1")
        group.addParam('box_size_z', params.FloatParam, label='Box size Z',
                      help="Box size along the z dimension", condition="boundary==1")

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
                      help="Target pressure in the NPT ensemble", condition="ensemble==2")

        group = form.addGroup('Contraints', condition="simulationType==1 or simulationType==3")
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

        # Volumes
        group = form.addGroup('Volume Parameters', condition="EMfitChoice==1")
        group.addParam('inputVolume', params.PointerParam, pointerClass="Volume, SetOfVolumes",
                      label="Input volume (s)", help='Select the target EM density volume',
                      condition="EMfitChoice==1", important=True)
        group.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="Voxel size in ANgstrom of the target volume", condition="EMfitChoice==1")
        group.addParam('centerOrigin', params.BooleanParam, label="Center Origin", default=True,
                      help="Center the volume to the origin", condition="EMfitChoice==1")
        group.addParam('origin_x', params.FloatParam, default=0, label="Origin X",
                      help="Origin of the first voxel in X direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")
        group.addParam('origin_y', params.FloatParam, default=0, label="Origin Y",
                      help="Origin of the first voxel in Y direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")
        group.addParam('origin_z', params.FloatParam, default=0, label="Origin Z",
                      help="Origin of the first voxel in Z direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")

        # Images
        group = form.addGroup('Image Parameters', condition="EMfitChoice==2")
        group.addParam('inputImage', params.PointerParam, pointerClass="Particle, SetOfParticles",
                      label="Input image (s)", help='Select the target EM density map',
                      condition="EMfitChoice==2", important=True)
        group.addParam('image_size', params.IntParam, default=64, label='Image Size',
                      help="TODO", condition="EMfitChoice==2")

        group.addParam('imageAngleShift', params.FileParam, label="Rigid body parameters (.xmd)",
                      condition="EMfitChoice==2",
                      help='Xmipp metadata file of rigid body parameters for each image (3 euler angles, 2 shift)')
        group.addParam('pixel_size', params.FloatParam, default=1.0, label='Pixel size (A)',
                      help="Pixel size of the EM data in Angstrom", condition="EMfitChoice==2")

        form.addParallelSection(threads=1, mpi=1)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Create INP files
        self._insertFunctionStep("createINPs")

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
            self.getNumberOfSimulation() >1 :
            numMpiPerFit, numLinearFit, numParallelFit, numLastIter = self.getMPIParams()
            for i in range(numLinearFit+1):
                self._insertFunctionStep("runSimulationParallel", i)
        else:
            for i in range(self.getNumberOfSimulation()):
                self._insertFunctionStep("runSimulation", i)

        # Create output data
        self._insertFunctionStep("createOutputStep")

    # --------------------------- Convert Input PDBs --------------------------------------------

    def convertInputPDBStep(self):
        """
        Convert input PDB step. Generate topology files and copy input PDB files
        :return None:
        """

        # COPY PDBS -------------------------------------------------------------
        inputPDBfn = self.getInputPDBfn()
        n_pdb = self.getNumberOfInputPDB()
        for i in range(n_pdb):
            runCommand("cp %s %s.pdb"%(inputPDBfn[i],self.getInputPDBprefix(i)))

        # TOPOLOGY FILES -------------------------------------------------
        if self.restartChoice.get():
            if self.getForceField() == FORCEFIELD_CHARMM:
                for i in range(n_pdb):
                    runCommand("cp %s.psf %s.psf" % (self.restartProt.get().getInputPDBprefix(i), self.getInputPDBprefix(i)))
            elif self.getForceField() == FORCEFIELD_AAGO  or self.getForceField() == FORCEFIELD_CAGO:
                for i in range(n_pdb):
                    runCommand("cp %s.top %s.top" % (self.restartProt.get().getInputPDBprefix(i), self.getInputPDBprefix(i)))
        else:
            if self.generateTop.get():
                #CHARMM
                if self.getForceField() == FORCEFIELD_CHARMM:
                    for i in range(n_pdb):
                        prefix = self.getInputPDBprefix(i)
                        generatePSF(inputPDB=prefix+".pdb",inputTopo=self.inputRTF.get(),
                            outputPrefix=prefix, nucleicChoice=self.nucleicChoice.get())
                # GO MODELS
                elif self.getForceField() == FORCEFIELD_AAGO or self.getForceField() == FORCEFIELD_CAGO:
                    for i in range(n_pdb):
                        prefix = self.getInputPDBprefix(i)
                        generatePSF(inputPDB=prefix+".pdb", inputTopo=self.inputRTF.get(),
                                    outputPrefix=prefix+"_AA", nucleicChoice=self.nucleicChoice.get())
                        generateGROTOP(inputPDB=prefix+"_AA.pdb", outputPrefix=prefix,
                                       forcefield=self.getForceField(), smog_dir=self.smog_dir.get(),
                        nucleicChoice=self.nucleicChoice.get())
            else:
                # CHARMM
                if self.getForceField() == FORCEFIELD_CHARMM:
                    for i in range(n_pdb):
                        runCommand("cp %s %s.psf" % (self.inputPSF.get(), self.getInputPDBprefix(i)))

                # GO MODELS
                elif self.getForceField() == FORCEFIELD_AAGO or self.getForceField() == FORCEFIELD_CAGO:
                    runCommand("cp %s %s.top" % (self.inputTOP.get(), self.getInputPDBprefix(i)))

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
        with open(nm_file, "w") as f:
            for i in range(self.inputModes.get().getSize()):
                if i >= 6:
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

        inputEMfn = self.getInputEMfn()
        n_em = self.getNumberOfInputEM()

        dest_ext = "mrc" if self.EMfitChoice.get() == EMFIT_VOLUMES else "spi"

        # Convert / copy EM data
        for i in range(n_em):
            pre, ext = os.path.splitext(os.path.basename(inputEMfn[i]))
            if self.EMfitChoice.get() == EMFIT_IMAGES and  ext == ".%s"%dest_ext:
                createLink(inputEMfn[i], "%s.%s" % (self.getInputEMprefix(i), dest_ext))
            else:
                runProgram("xmipp_image_convert", "-i %s --oext %s -o %s.%s" %
                           (inputEMfn[i], dest_ext, self.getInputEMprefix(i), dest_ext))

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

    def createINPs(self):
        """
        Create GENESIS input files
        :return None:
        """
        for indexFit in range(self.getNumberOfSimulation()):
            outputPrefix = self.getOutputPrefix(indexFit)
            inputPDBprefix = self.getInputPDBprefix(indexFit)
            inputEMprefix = self.getInputEMprefix(indexFit)
            inp_file = self._getExtraPath("%s_INP" % str(indexFit + 1).zfill(5))
            if self.restartChoice.get():
                inputProt = self.restartProt.get()
            else:
                inputProt = self

            s = "\n[INPUT] \n"  # -----------------------------------------------------------
            s += "pdbfile = %s.pdb\n" % inputPDBprefix
            if self.getForceField() == FORCEFIELD_CHARMM:
                s += "topfile = %s\n" % inputProt.inputRTF.get()
                s += "parfile = %s\n" % inputProt.inputPRM.get()
                s += "psffile = %s.psf\n" % inputPDBprefix
                if inputProt.inputSTR.get() != "" and inputProt.inputSTR.get() is not None:
                    s += "strfile = %s\n" % inputProt.inputSTR.get()
            elif self.getForceField() == FORCEFIELD_AAGO or self.getForceField() == FORCEFIELD_CAGO:
                s += "grotopfile = %s.top\n" % inputPDBprefix
            if self.restartChoice.get():
                s += "rstfile = %s \n" % self.getRestartFile(indexFit)

            s += "\n[OUTPUT] \n"  # -----------------------------------------------------------
            if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
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
            if self.getForceField() == FORCEFIELD_CHARMM:
                s += "forcefield = CHARMM \n"
            elif self.getForceField() == FORCEFIELD_AAGO:
                s += "forcefield = AAGO  \n"
            elif self.getForceField() == FORCEFIELD_CAGO:
                s += "forcefield = CAGO  \n"

            if self.electrostatics.get() == ELECTROSTATICS_CUTOFF:
                s += "electrostatic = CUTOFF  \n"
            else:
                s += "electrostatic = PME  \n"
            s += "switchdist   = %.2f \n" % self.switch_dist.get()
            s += "cutoffdist   = %.2f \n" % self.cutoff_dist.get()
            s += "pairlistdist = %.2f \n" % self.pairlist_dist.get()
            if self.vdw_force_switch.get():
                s += "vdw_force_switch = YES \n"
            if self.implicitSolvent.get() == IMPLICIT_SOLVENT_GBSA:
                s += "implicit_solvent = GBSA \n"
                s += "gbsa_eps_solvent = 78.5 \n"
                s += "gbsa_eps_solute  = 1.0 \n"
                s += "gbsa_salt_cons   = 0.2 \n"
                s += "gbsa_surf_tens   = 0.005 \n"

            if self.simulationType.get() == SIMULATION_MIN:
                s += "\n[MINIMIZE]\n"  # -----------------------------------------------------------
                s += "method = SD\n"
            else:
                s += "\n[DYNAMICS] \n"  # -----------------------------------------------------------
                if self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD:
                    s += "integrator = NMMD  \n"
                elif self.integrator.get() == INTEGRATOR_VVERLET:
                    s += "integrator = VVER  \n"
                elif self.integrator.get() == INTEGRATOR_LEAPFROG:
                    s += "integrator = LEAP  \n"

                s += "timestep = %f \n" % self.time_step.get()
            s += "nsteps = %i \n" % self.n_steps.get()
            s += "eneout_period = %i \n" % self.eneout_period.get()
            s += "crdout_period = %i \n" % self.crdout_period.get()
            s += "rstout_period = %i \n" % self.n_steps.get()
            s += "nbupdate_period = %i \n" % self.nbupdate_period.get()

            if self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD:
                s += "\n[NMMD] \n"  # -----------------------------------------------------------
                s += "nm_number = %i \n" % self.nm_number.get()
                s += "nm_mass = %f \n" % self.nm_mass.get()
                s += "nm_file = %s.nma \n" % inputPDBprefix
                if self.nm_init.get() is not None and self.nm_init.get() != "":
                    s += "nm_init = %s \n" % " ".join([str(i) for i in np.loadtxt(self.nm_init.get())[indexFit]])
                if self.nm_dt.get() is None:
                    s += "nm_dt = %f \n" % self.time_step.get()
                else:
                    s += "nm_dt = %f \n" % self.nm_dt.get()

            if self.simulationType.get() != SIMULATION_MIN:
                s += "\n[CONSTRAINTS] \n"  # -----------------------------------------------------------
                if self.rigid_bond.get():
                    s += "rigid_bond = YES \n"
                else:
                    s += "rigid_bond = NO \n"
                if self.fast_water.get():
                    s += "fast_water = YES \n"
                    s += "water_model = %s \n" % self.water_model.get()
                else:
                    s += "fast_water = NO \n"

            s += "\n[BOUNDARY] \n"  # -----------------------------------------------------------
            if self.boundary.get() == BOUNDARY_PBC:
                s += "type = PBC \n"
                s += "box_size_x = %f \n" % self.box_size_x.get()
                s += "box_size_y = %f \n" % self.box_size_y.get()
                s += "box_size_z = %f \n" % self.box_size_z.get()
            else:
                s += "type = NOBC \n"

            if self.simulationType.get() != SIMULATION_MIN:
                s += "\n[ENSEMBLE] \n"  # -----------------------------------------------------------
                if self.ensemble.get() == ENSEMBLE_NVE:
                    s += "ensemble = NVE  \n"
                elif self.ensemble.get() == ENSEMBLE_NPT:
                    s += "ensemble = NPT  \n"
                else:
                    s += "ensemble = NVT  \n"
                if self.tpcontrol.get() == TPCONTROL_LANGEVIN:
                    s += "tpcontrol = LANGEVIN  \n"
                elif self.tpcontrol.get() == TPCONTROL_BERENDSEN:
                    s += "tpcontrol = BERENDSEN  \n"
                elif self.tpcontrol.get() == TPCONTROL_BUSSI:
                    s += "tpcontrol = BUSSI  \n"
                else:
                    s += "tpcontrol = NO  \n"
                s += "temperature = %.2f \n" % self.temperature.get()
                if self.ensemble.get() == ENSEMBLE_NPT:
                    s += "pressure = %.2f \n" % self.pressure.get()

            if (self.EMfitChoice.get() == EMFIT_VOLUMES or self.EMfitChoice.get() == EMFIT_IMAGES) \
                    and self.simulationType.get() != SIMULATION_MIN:
                s += "\n[SELECTION] \n"  # -----------------------------------------------------------
                s += "group1 = all and not hydrogen\n"

                s += "\n[RESTRAINTS] \n"  # -----------------------------------------------------------
                s += "nfunctions = 1 \n"
                s += "function1 = EM \n"
                constStr = self.constantK.get()
                if "-" in constStr:
                    splt = constStr.split("-")
                    constStr = " ".join(
                        [str(int(i)) for i in np.linspace(int(splt[0]), int(splt[1]), self.nreplica.get())])
                s += "constant1 = %s \n" % constStr
                s += "select_index1 = 1 \n"

                s += "\n[EXPERIMENTS] \n"  # -----------------------------------------------------------
                s += "emfit = YES  \n"
                s += "emfit_sigma = %.4f \n" % self.emfit_sigma.get()
                s += "emfit_tolerance = %.6f \n" % self.emfit_tolerance.get()
                s += "emfit_period = 1  \n"
                if self.EMfitChoice.get() == EMFIT_VOLUMES:
                    s += "emfit_target = %s.mrc \n" % inputEMprefix
                elif self.EMfitChoice.get() == EMFIT_IMAGES:
                    s += "emfit_type = IMAGE \n"
                    s += "emfit_target = %s.spi \n" % inputEMprefix
                    s += "emfit_pixel_size =  %f\n" % self.pixel_size.get()
                    rigid_body_params = self.getRigidBodyParams(indexFit)
                    s += "emfit_roll_angle = %f\n" % rigid_body_params[0]
                    s += "emfit_tilt_angle = %f\n" % rigid_body_params[1]
                    s += "emfit_yaw_angle =  %f\n" % rigid_body_params[2]
                    s += "emfit_shift_x = %f\n" % rigid_body_params[3]
                    s += "emfit_shift_y =  %f\n" % rigid_body_params[4]

                if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
                    s += "\n[REMD] \n"  # -----------------------------------------------------------
                    s += "dimension = 1 \n"
                    s += "exchange_period = %i \n" % self.exchange_period.get()
                    s += "type1 = RESTRAINT \n"
                    s += "nreplica1 = %i \n" % self.nreplica.get()
                    s += "rest_function1 = 1 \n"

            with open(inp_file, "w") as f:
                f.write(s)

    def runSimulation(self, index):
        """
        Run GENESIS simulations
        :return None:
        """
        programname = "atdyn" if self.md_program.get() == PROGRAM_ATDYN else "spdyn"
        inp_file =self._getExtraPath("%s_INP" % str(index + 1).zfill(5))
        params = "%s > %s.log" % (inp_file,self.getOutputPrefix(index))
        env = self.getGenesisEnv()
        env.set("OMP_NUM_THREADS",str(self.numberOfThreads.get()))

        self.runJob(programname,params, env=env)

    def runSimulationParallel(self, indexLinearFit):
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
        extradir = self._getExtraPath()
        params = "%s/{}_INP > %s/{}_output.log " %(extradir, extradir)
        cmd = buildRunCommand(programname, params, numberOfMpi=numberOfMpiPerFit, hostConfig=self._stepsExecutor.hostConfig,
                              env=env)

        # Build parallel command
        parallel_cmd = "seq -f \"%%05g\" 1 %i | parallel -P %i \" %s\" " % (
        self.getNumberOfSimulation(),self.numberOfMpi.get()//numberOfMpiPerFit, cmd)

        print("Command : %s" % cmd)
        print("Parallel Command : %s" % parallel_cmd)
        runCommand(parallel_cmd, env=env)


    # --------------------------- Create output step --------------------------------------------

    def createOutputStep(self):
        """
        Create output PDB or set of PDBs
        :return None:
        """
        if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
            self.convertReusOutputDcd()

        # Convert Output
        for i in range(self.getNumberOfSimulation()):
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                # Extract the pdb from the DCD file in case of SPDYN
                if self.md_program.get() == PROGRAM_SPDYN:
                    lastPDBFromDCD(
                        inputDCD=j + ".dcd",
                        outputPDB=j + ".pdb",
                        inputPDB=self.getInputPDBprefix(i) + ".pdb")


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

            pdbset = self._createSetOfPDBs("outputPDBs")
            # Add each output PDB to the Set
            for i in range(self.getNumberOfSimulation()):
                outputPrefix =self.getOutputPrefixAll(i)
                for j in outputPrefix:
                    pdbset.append(AtomStruct(j + ".pdb"))
            self._defineOutputs(outputPDBs=pdbset)

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
        return ["kobayashi2017genesis","vuillemot2022NMMD"]

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------

    def getNumberOfInputPDB(self):
        """
        Get the number of input PDBs
        :return int: number of input PDBs
        """
        if self.restartChoice.get():
            allOutPrx = []
            for i in range(self.restartProt.get().getNumberOfSimulation()):
                allOutPrx += self.restartProt.get().getOutputPrefixAll(i)
            return len(allOutPrx )
        else:
            if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                    isinstance(self.inputPDB.get(), SetOfPDBs):
                return self.inputPDB.get().getSize()
            else: return 1

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

    def getInputPDBfn(self):
        """
        Get the input PDB file names
        :return list : list of input PDB file names
        """
        initFn = []

        if self.restartChoice.get():
            for i in range(self.restartProt.get().getNumberOfSimulation()):
                initFn += self.restartProt.get().getOutputPrefixAll(i)
            initFn = [i+".pdb" for i in initFn]
        else:
            if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                    isinstance(self.inputPDB.get(), SetOfPDBs):
                for i in range(self.inputPDB.get().getSize()):
                    initFn.append(self.inputPDB.get()[i+1].getFileName())

            else:
                initFn.append(self.inputPDB.get().getFileName())
        return initFn

    def getInputEMfn(self):
        """
        Get the input EM data file names
        :return list: list of input EM data file names
        """
        inputEMfn = []
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), SetOfVolumes) :
                for i in self.inputVolume.get():
                    inputEMfn.append(i.getFileName())
            else:
                inputEMfn.append(self.inputVolume.get().getFileName())
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            if isinstance(self.inputImage.get(), SetOfParticles) :
                for i in self.inputImage.get():
                    inputEMfn.append(i.getFileName())
            else:
                inputEMfn.append(self.inputImage.get().getFileName())
        return inputEMfn

    def getInputPDBprefix(self, index=0):
        """
        Get the input PDB prefix of the specified index
        :param int index: index of input PDB
        :return str: Input PDB prefix
        """
        prefix = self._getExtraPath("%s_inputPDB")
        if self.getNumberOfInputPDB() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)

    def getInputEMprefix(self, index=0):
        """
        Get the input EM data prefix of the specified index
        :param int index: index of the EM data
        :return str: Input EM data prefix
        """
        prefix = self._getExtraPath("%s_inputEM")
        if self.getNumberOfInputEM() == 0:
            return ""
        elif self.getNumberOfInputEM() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)


    def getOutputPrefix(self, index=0):
        """
        Output prefix of the specified index
        :param int index: index of the simulation to get
        :return string : Output prefix of the specified index
        """
        return self._getExtraPath("%s_output" % str(index + 1).zfill(5))

    def getOutputPrefixAll(self, index=0):
        """
        All output prefix of the specified index including multiple replicas in case of REUS
        :param int index: index of the simulation to get
        :return list: list of all output prefix of the specified index
        """
        outputPrefix=[]
        if self.simulationType.get() == SIMULATION_REMD or self.simulationType.get() == SIMULATION_RENMMD:
            for i in range(self.nreplica.get()):
                outputPrefix.append(self._getExtraPath("%s_output_remd%i" %
                                (str(index + 1).zfill(5), i + 1)))
        else:
            outputPrefix.append(self._getExtraPath("%s_output" % str(index + 1).zfill(5)))
        return outputPrefix

    def getRigidBodyParams(self, index=0):
        """
        Get the current rigid body parameters for the specified index in case of EMFIT with iamges
        :param int index: Index of the simulation
        :return list: angle_rot, angle_tilt, angle_psi, shift_x, shift_y
        """
        mdImg = md.MetaData(self.imageAngleShift.get())
        idx = int(index + 1)
        params = [
            mdImg.getValue(md.MDL_ANGLE_ROT, idx),
            mdImg.getValue(md.MDL_ANGLE_TILT, idx),
            mdImg.getValue(md.MDL_ANGLE_PSI, idx),
            mdImg.getValue(md.MDL_SHIFT_X, idx),
            mdImg.getValue(md.MDL_SHIFT_Y, idx),
        ]
        return params

    def getGenesisEnv(self):
        """
        Get environnement for running GENESIS
        :return Environ: environnement
        """
        environ = pwutils.Environ(os.environ)
        environ.set('PATH', os.path.join(Plugin.getVar("GENESIS_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        return environ

    def getRestartFile(self, index=0):
        """
        Get input restart file
        :param int index: Index of the simulation
        :return str: restart file
        """
        allOutPrx = []
        for i in range(self.restartProt.get().getNumberOfSimulation()):
            allOutPrx += self.restartProt.get().getOutputPrefixAll(i)
        allOut = [i + ".rst" for i in allOutPrx]
        return allOut[int(np.min([len(allOut)-1, index]))]

    def getForceField(self):
        """
        Get simulation forcefield
        :return int: forcefield
        """
        if self.restartChoice.get():
            return self.restartProt.get().getForceField()
        else:
            return self.forcefield.get()


    def getMPIParams(self):
        """
        Get mpi parameters for the simulation
        :return tuple: numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter
        """
        nreplica = 1
        n_fit = self.getNumberOfSimulation() * nreplica

        if n_fit <= self.numberOfMpi.get():
            numberOfMpiPerFit   = self.numberOfMpi.get()//self.getNumberOfSimulation()
            numberOfLinearFit   = 1
            numberOfParallelFit = self.getNumberOfSimulation()
            numberOflastIter    = 0
        else:
            numberOfMpiPerFit   = nreplica
            numberOfLinearFit   = n_fit//self.numberOfMpi.get()
            numberOfParallelFit = self.numberOfMpi.get()//nreplica
            numberOflastIter    = n_fit % self.numberOfMpi.get()

        return numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter

    def convertReusOutputDcd(self):

        for i in range(self.getNumberOfSimulation()):
            remdPrefix = self._getExtraPath("%s_output_remd" % str(i + 1).zfill(5))
            tmpPrefix =  self._getExtraPath("%s_output_tmp" % str(i + 1).zfill(5))
            inp_file = self._getExtraPath("tmp_INP")

            with open(inp_file, "w") as f:
                f.write("\n[INPUT]\n")
                f.write("reffile = %s.pdb # PDB file\n" % self.getInputPDBprefix(i))
                f.write("remfile = %s{}.rem  # REMD parameter ID file\n" % remdPrefix)
                f.write("dcdfile = %s{}.dcd  # DCD file\n" % remdPrefix)
                f.write("logfile = %s{}.log  # REMD energy log file\n" % remdPrefix)

                f.write("\n[OUTPUT]\n")
                f.write("trjfile = %s{}.dcd  # coordinates sorted by temperature\n"% tmpPrefix)
                f.write("logfile = %s{}.log  # energy log sorted by temperature\n"% tmpPrefix)

                f.write("\n[SELECTION]\n")
                f.write("group1 = all  # selection group 1\n")

                f.write("\n[FITTING]\n")
                f.write("fitting_method = NO  # [NO,TR,TR+ROT,TR+ZROT,XYTR,XYTR+ZROT]\n")
                f.write("mass_weight = NO  # mass-weight is not applied\n")

                f.write("\n[OPTION]\n")
                f.write("check_only = NO\n")
                f.write("convert_type = PARAMETER  # (REPLICA/PARAMETER)\n")
                f.write("num_replicas = %i  # total number of replicas used in the simulation\n"% self.nreplica.get())
                f.write("convert_ids =  # selected index (empty = all)(example: 1 2 5-10)\n")
                f.write("nsteps = %i  # nsteps in [DYNAMICS]\n" % self.n_steps.get())
                f.write("exchange_period = %i  # exchange_period in [REMD]\n" % self.exchange_period.get())
                f.write("crdout_period = %i  # crdout_period in [DYNAMICS]\n" % self.eneout_period.get() )
                f.write("eneout_period = %i  # eneout_period in [DYNAMICS]\n" % self.crdout_period.get() )
                f.write("trjout_format = DCD  # (PDB/DCD)\n")
                f.write("trjout_type = COOR+BOX  # (COOR/COOR+BOX)\n")
                f.write("trjout_atom = 1  # atom group\n")
                f.write("centering = NO\n")
                f.write("pbc_correct = NO\n")

            runCommand("remd_convert %s"%inp_file, env=self.getGenesisEnv())
            for j in range(self.nreplica.get()):
                repPrefix = self._getExtraPath("%s_output_remd%i" % (str(i + 1).zfill(5), j+1))
                reptmpPrefix = self._getExtraPath("%s_output_tmp%i" % (str(i + 1).zfill(5), j+1))
                runCommand("mv %s.dcd %s.dcd"%(reptmpPrefix,repPrefix))
                runCommand("mv %s.log %s.log"%(reptmpPrefix,repPrefix))
