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
from pwem.protocols import EMProtocol
from pwem.objects.data import AtomStruct, SetOfAtomStructs, SetOfPDBs, SetOfVolumes,SetOfParticles

import numpy as np
import mrcfile
from pwem.utils import runProgram

from .utilities.genesis_utilities import *
from xmipp3 import Plugin
import pyworkflow.utils as pwutils
from pyworkflow.utils import runCommand

class ProtGenesis(EMProtocol):
    """ Protocol to perform MD simulation using GENESIS. """
    _label = 'Genesis'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        # Inputs ============================================================================================
        form.addSection(label='Inputs')
        form.addParam('md_program', params.EnumParam, label="MD program", default=PROGRAM_ATDYN,
                      choices=['ATDYN', 'SPDYN'],
                      help="SPDYN (Spatial decomposition dynamics) and ATDYN (Atomic decomposition dynamics)"
                    " share almost the same data structures, subroutines, and modules, but differ in"
                    " their parallelization schemes. In SPDYN, the spatial decomposition scheme is implemented with new"
                    " parallel algorithms and GPGPU calculation. In ATDYN, the atomic decomposition scheme"
                    " is introduced for simplicity. The performance of ATDYN is not comparable to SPDYN due to the"
                    " simple parallelization scheme but contains new methods and features.", important=True,
                      expertLevel=params.LEVEL_ADVANCED)

        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct, SetOfPDBs, SetOfAtomStructs', label="Input PDB (s)",
                      help='Select the input PDB or set of PDBs.', important=True)

        form.addParam('inputRST', params.FileParam, label="GENESIS Restart File (optional)",
                       help='Restart a previous GENESIS run with a .rst file', default="")

        group = form.addGroup('Forcefield Inputs')
        group.addParam('forcefield', params.EnumParam, label="Forcefield type", default=0, important=True,
                      choices=['CHARMM', 'AAGO', 'CAGO'], help="Type of the force field used for energy and force calculation")
        group.addParam('generateTop', params.BooleanParam, label="Generate topology files ?",
                      default=False, help="Use the GUI to generate topology files for you (PSF file for CHARMM and TOP file for AAGO/CAGO)."
                                          " Requires VMD psfgen for CHARMM forcefields "
                                          " and SMOG2 for GO models. Note that the generated topology files will not include"
                                          " solvent.")
        group.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=0,
                      choices=['NO', 'RNA', 'DNA'], condition ="generateTop",help="TODo")
        group.addParam('smog_dir', params.FileParam, label="SMOG2 directory",
                      help='Path to SMOG2 directory', condition="(forcefield==1 or forcefield==2) and generateTop")
        group.addParam('inputTOP', params.FileParam, label="GROMACS Topology File",
                      condition="(forcefield==1 or forcefield==2) and not generateTop",
                      help='Gromacs ‘top’ file containing information of the system such as atomic masses, charges,'
                           ' atom connectivities. For details about this format, see the Gromacs web site')
        group.addParam('inputPRM', params.FileParam, label="CHARMM parameter file",
                      condition = "forcefield==0",
                      help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries' )
        group.addParam('inputRTF', params.FileParam, label="CHARMM topology file",
                      condition="forcefield==0 or ((forcefield==1 or forcefield==2) and generateTop)",
                      help='CHARMM topology file containing information about atom connectivity of residues and'
                           ' other molecules. For details on the format, see the CHARMM web site.'
                           ' In the case of generating topology files for GO models, '
                           ' the CHARMM topology file and VMD psfgen are used to fill missing atoms/residues.')
        group.addParam('inputPSF', params.FileParam, label="CHARMM Structure File",
                      condition="forcefield==0 and not generateTop",
                      help='CHARMM/X-PLOR psf file containing information of the system such as atomic masses,'
                            ' charges, and atom connectivities. To generate this file, you can either use the option'
                           '\" generate topology files\", VMD psfgen, or online CHARMM GUI.')
        group.addParam('inputSTR', params.FileParam, label="CHARMM stream file (optional)",
                      condition="forcefield==0", default="",
                      help='CHARMM stream file containing both topology information and parameters')



        # Simulation =================================================================================================
        form.addSection(label='Simulation')
        form.addParam('simulationType', params.EnumParam, label="Simulation type", default=0,
                      choices=['Molecular Dynamics', 'Minimization', 'Replica-Exchange Molecular Dynamics'],
                      help="Type of simulation to be performed by GENESIS", important=True)
        form.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['Velocity Verlet', 'Leapfrog', 'NMMD'],
                      help="Type of integrator for the MD simulation", condition="simulationType!=1")
        form.addParam('time_step', params.FloatParam, default=0.002, label='Time step (ps)',
                      help="Time step in the MD run", condition="simulationType!=1")
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Total number of steps in one MD run")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="Output frequency for the energy data")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinate output period',
                      help="Output frequency for the coordinates data")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="Update frequency of the non-bonded pairlist",
                      expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('NMMD parameters', condition="integrator==2 and simulationType!=1")
        group.addParam('nm_number', params.IntParam, default=10, label='Number of normal modes',
                      help="Number of normal modes for NMMD. 10 should work in most cases. Avoid "
                           " using too much NM (>50).",
                      condition="integrator==2 and simulationType!=1")
        group.addParam('nm_mass', params.FloatParam, default=10.0, label='NM mass',
                      help="Mass value of Normal modes for NMMD", condition="integrator==2 and simulationType!=1",
                      expertLevel=params.LEVEL_ADVANCED)
        group.addParam('nm_limit', params.FloatParam, default=1000.0, label='NM amplitude threshold',
                      help="Threshold of normal mode amplitude above which the normal modes are updated",
                      condition="integrator==2 and simulationType!=1",expertLevel=params.LEVEL_ADVANCED)
        group.addParam('elnemo_cutoff', params.FloatParam, default=8.0, label='NMA cutoff (A)',
                      help="Cutoff distance for elastic network model", condition="integrator==2 and simulationType!=1",
                      expertLevel=params.LEVEL_ADVANCED)
        group.addParam('elnemo_rtb_block', params.IntParam, default=10, label='NMA Number of residue RTB',
                      help="Number of residue per RTB block in the NMA computation",
                      condition="integrator==2 and simulationType!=1",expertLevel=params.LEVEL_ADVANCED)

        group = form.addGroup('REMD parameters', condition="simulationType==2")
        group.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="Number of MD steps between replica exchanges", condition="simulationType==2")
        group.addParam('nreplica', params.IntParam, default=1, label='Number of replicas',
                      help="Number of replicas for REMD", condition="simulationType==2")
        # ENERGY =================================================================================================
        form.addSection(label='Energy')
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['GBSA', 'NONE'],
                      help="Turn on Generalized Born/Solvent accessible surface area model. Boundary condition must be NO."
                           " ATDYN only.")

        form.addParam('electrostatics', params.EnumParam, label="Non-bonded interactions", default=1,
                      choices=['PME', 'Cutoff'],
                      help="Type of Non-bonded interactions. "
                           " CUTOFF: Non-bonded interactions including the van der Waals interaction are just"
                           " truncated at cutoffdist; "
                           " PME : Particle mesh Ewald (PME) method is employed for long-range interactions."
                            " This option is only availabe in the periodic boundary condition")
        form.addParam('vdw_force_switch', params.BooleanParam, label="Switch function Van der Waals", default=True,
                      help="This paramter determines whether the force switch function for van der Waals interactions is"
                        " employed or not. The users must take care about this parameter, when the CHARMM"
                        " force field is used. Typically, vdw_force_switch=YES should be specified in the case of"
                        " CHARMM36",expertLevel=params.LEVEL_ADVANCED)
        form.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance',
                      help="Switch-on distance for nonbonded interaction energy/force quenching")
        form.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance',
                      help="Cut-off distance for the non-bonded interactions. This distance must be larger than"
                            " switchdist, while smaller than pairlistdist")
        form.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance',
                      help="Distance used to make a Verlet pair list for non-bonded interactions . This distance"
                            " must be larger than cutoffdist")

        # Ensemble =================================================================================================
        form.addSection(label='Ensemble')
        form.addParam('ensemble', params.EnumParam, label="Ensemble", default=0,
                      choices=['NVT', 'NVE', 'NPT'],
                      help="Type of ensemble, NVE: Microcanonical ensemble, NVT: Canonical ensemble,"
                           " NPT: Isothermal-isobaric ensemble")
        form.addParam('tpcontrol', params.EnumParam, label="Thermostat/Barostat", default=1,
                      choices=['NO', 'LANGEVIN', 'BERENDSEN', 'BUSSI'],
                      help="Type of thermostat and barostat. The availabe algorithm depends on the integrator :"
                           " LEAP : BERENDSEN, LANGEVIN;  VVER : BERENDSEN (NVT only), LANGEVIN, BUSSI; "
                           " NMMD : LANGEVIN (NVT only)")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="Initial and target temperature")
        form.addParam('pressure', params.FloatParam, default=1.0, label='Pressure (atm)',
                      help="Target pressure in the NPT ensemble", condition="ensemble==2")
        # Boundary =================================================================================================
        form.addSection(label='Boundary')
        form.addParam('boundary', params.EnumParam, label="Boundary", default=0,
                      choices=['No boundary', 'Periodic Boundary Condition'],
                      help="Type of boundary condition")
        form.addParam('box_size_x', params.FloatParam, label='Box size X',
                      help="Box size along the x dimension", condition="boundary==1")
        form.addParam('box_size_y', params.FloatParam, label='Box size Y',
                      help="Box size along the y dimension", condition="boundary==1")
        form.addParam('box_size_z', params.FloatParam, label='Box size Z',
                      help="Box size along the z dimension", condition="boundary==1")
        # Experiments =================================================================================================
        form.addSection(label='Experiments')
        form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=0,
                      choices=['None', 'Volume (s)', 'Image (s)'], important=True,
                      help="Type of cryo-EM data to be processed")
        form.addParam('centerPDB', params.BooleanParam, label="Center PDB ?",
                      default=False, help="Center the input PDBs with the center of mass")
        form.addParam('constantK', params.StringParam, default="10000", label='Force constant (kcal/mol)',
                      help="Force constant in Eem = k*(1 - c.c.). Note that in the case of REUS, the number of "
                           " force constant value must be equal to the number of replicas, for example for 4 replicas,"
                           " a valid force constant is \"1000 2000 3000 4000\" "
                      , condition="EMfitChoice!=0")
        form.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EMfit Sigma",
                      help="Resolution parameter of the simulated map. This is usually set to the half of the resolution"
                        " of the target map. For example, if the target map resolution is 5 Å, emfit_sigma=2.5",
                      condition="EMfitChoice!=0",expertLevel=params.LEVEL_ADVANCED)
        form.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EMfit Tolerance',
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
        group.addParam('origin_x', params.FloatParam, default=None, label="Origin X",
                      help="Origin of the first voxel in X direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")
        group.addParam('origin_y', params.FloatParam, default=None, label="Origin X",
                      help="Origin of the first voxel in X direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")
        group.addParam('origin_z', params.FloatParam, default=None, label="Origin X",
                      help="Origin of the first voxel in X direction (in Angstrom) ",
                      condition="EMfitChoice==1 and not centerOrigin")

        # Constraints =================================================================================================
        form.addSection(label='Constraints')
        form.addParam('rigid_bond', params.BooleanParam, label="Rigid bonds (SHAKE/RATTLE)",
                      default=False,
                      help="Turn on or off the SHAKE/RATTLE algorithms for covalent bonds involving hydrogen")
        form.addParam('fast_water', params.BooleanParam, label="Fast water (SETTLE)",
                      default=False,
                      help="Turn on or off the SETTLE algorithm for the constraints of the water molecules")
        form.addParam('water_model', params.StringParam, label='Water model', default="TIP3",
                      help="Residue name of the water molecule to be rigidified in the SETTLE algorithm", condition="fast_water")

        form.addParallelSection(threads=1, mpi=1)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputPDBStep")
        if self.EMfitChoice.get() != EMFIT_NONE:
            self._insertFunctionStep("convertInputEMStep")
        self._insertFunctionStep("runGenesisStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- Convert Input PDBs --------------------------------------------

    def convertInputPDBStep(self):
        """
        Convert input PDB step. Generate topology files and copy input PDB files
        :return None:
        """

        inputPDBfn = self.getInputPDBfn()
        n_pdb = self.getNumberOfInputPDB()

        # Copy PDBs :
        for i in range(n_pdb):
            runCommand("cp %s %s.pdb"%(inputPDBfn[i],self.getInputPDBprefix(i)))

        # GENERATE TOPOLOGY FILES
        if self.generateTop.get():
            #CHARMM
            if self.forcefield.get() == FORCEFIELD_CHARMM:
                for i in range(n_pdb):
                    prefix = self.getInputPDBprefix(i)
                    generatePSF(inputPDB=prefix+".pdb",inputTopo=self.inputRTF.get(),
                        outputPrefix=prefix, nucleicChoice=self.nucleicChoice.get())

            # GROMACS
            elif self.forcefield.get() == FORCEFIELD_AAGO\
                    or self.forcefield.get() == FORCEFIELD_CAGO:
                self.inputTOPfn = []
                for i in range(n_pdb):
                    prefix = self.getInputPDBprefix(i)
                    generatePSF(inputPDB=prefix+".pdb", inputTopo=self.inputRTF.get(),
                                outputPrefix=prefix, nucleicChoice=self.nucleicChoice.get())
                    generateGROTOP(inputPDB=prefix+".pdb", outputPrefix=prefix,
                                   forcefield=self.forcefield.get(), smog_dir=self.smog_dir.get(),
					nucleicChoice=self.nucleicChoice.get())

        else:
            # CHARMM
            if self.forcefield.get() == FORCEFIELD_CHARMM:
                for i in range(n_pdb):
                    runCommand("cp %s %s.psf" % (self.inputPSF.get(), self.getInputPDBprefix(i)))

            # GROMACS
            elif self.forcefield.get() == FORCEFIELD_AAGO\
                    or self.forcefield.get() == FORCEFIELD_CAGO:
                runCommand("cp %s %s.top" % (self.inputTOP.get(), self.getInputPDBprefix(i)))

        # Center PDBs
        if self.centerPDB.get():
            for i in range(self.getNumberOfInputPDB()):
                cmd = "xmipp_pdb_center -i %s.pdb -o %s.pdb" %\
                        (self.getInputPDBprefix(i),self.getInputPDBprefix(i))
                runCommand(cmd)


    # --------------------------- Convert Input EM data --------------------------------------------

    def convertInputEMStep(self):
        """
        Convert EM data step
        :return None:
        """
        inputEMfn = self.getInputEMfn()
        n_em = self.getNumberOfInputEM()

        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            for i in range(n_em):
                self.convertInputVol(fnInput=inputEMfn[i], volPrefix = self.getInputEMprefix(i))

    def convertInputVol(self,fnInput,volPrefix):
        """
        Convert input volume data
        :param str fnInput: input volume file name
        :param str volPrefix: ouput volume prefix
        :return None:
        """

        # Convert data to mrc
        pre, ext = os.path.splitext(os.path.basename(fnInput))
        if ext != ".mrc":
            runProgram("xmipp_image_convert", "-i %s --oext mrc -o %s.mrc" %
                        (fnInput,volPrefix))
        else:
            runProgram("cp","%s %s.mrc" %(fnInput,volPrefix))

        # Update mrc header
        with mrcfile.open("%s.mrc" % volPrefix) as old_mrc:
            with mrcfile.new("%s.mrc" % volPrefix, overwrite=True) as new_mrc:
                new_mrc.set_data(old_mrc.data)
                new_mrc.voxel_size = self.voxel_size.get()
                new_mrc.header['origin'] = old_mrc.header['origin']
                if self.centerOrigin.get():
                    origin = -np.array(old_mrc.data.shape)/2 *self.voxel_size.get()
                    new_mrc.header['origin']['x'] = origin[0]
                    new_mrc.header['origin']['y'] = origin[1]
                    new_mrc.header['origin']['z'] = origin[2]
                else:
                    if self.origin_x.get() is not None:
                        new_mrc.header['origin']['x'] = self.origin_x.get()
                    if self.origin_y.get() is not None:
                        new_mrc.header['origin']['y'] = self.origin_y.get()
                    if self.origin_z.get() is not None:
                        new_mrc.header['origin']['z'] = self.origin_z.get()
                new_mrc.update_header_from_data()
                new_mrc.update_header_stats()

    # --------------------------- GENESIS step --------------------------------------------

    def runGenesisStep(self):
        """
        Run GENESIS simulations step
        :return None:
        """
        self.runParallelGenesis()


    def runParallelGenesis(self):
        """
        Run multiple GENESIS simulations in parallel
        :return None:
        """

        # SETUP MPI parameters
        numMpiPerFit, numLinearFit, numParallelFit, numLastIter = self.getMPIParams()

        for i1 in range(numLinearFit + 1):
            cmds = []
            n_parallel = numParallelFit if i1 < numLinearFit else numLastIter
            for i2 in range(n_parallel):
                indexFit = i2 + i1 * numParallelFit
                prefix = self.getOutputPrefix(indexFit)

                # Create INP file
                self.createINP(inputPDB=self.getInputPDBprefix(indexFit) + ".pdb",
                               outputPrefix=prefix, indexFit=indexFit)

                # Create Genesis command
                genesis_cmd = self.getGenesisCmd(prefix=prefix, n_mpi=numMpiPerFit)
                cmds.append(genesis_cmd)

            # Run Genesis
            runParallelJobs(cmds, env=self.getGenesisEnv(), numberOfMpi=numMpiPerFit,
                            numberOfThreads=self.numberOfThreads.get())

    def createINP(self,inputPDB, outputPrefix, indexFit):
        """
        Create INP input file for GENESIS
        :param str inputPDB: input PDB file name
        :param str outputPrefix: output prefix
        :param int indexFit: index of the simulation
        :return None:
        """
        inputPDBprefix = self.getInputPDBprefix(indexFit)
        inputEMprefix = self.getInputEMprefix(indexFit)
        inp_file = "%s_INP"% outputPrefix

        s = "\n[INPUT] \n" #-----------------------------------------------------------
        s += "pdbfile = %s\n" % inputPDB
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            s += "topfile = %s\n" % self.inputRTF.get()
            s += "parfile = %s\n" % self.inputPRM.get()
            s += "psffile = %s.psf\n" % inputPDBprefix
            if self.inputSTR.get() != "" and self.inputSTR.get() is not None:
                s += "strfile = %s\n" % self.inputSTR.get()
        elif self.forcefield.get() == FORCEFIELD_AAGO\
                or self.forcefield.get() == FORCEFIELD_CAGO:
            s += "grotopfile = %s.top\n" % inputPDBprefix
        if self.inputRST.get() != "" and self.inputRST.get() is not None:
            s += "rstfile = %s\n" % self.getRestartFile(indexFit)

        s += "\n[OUTPUT] \n" #-----------------------------------------------------------
        if self.simulationType.get() == SIMULATION_REMD:
            s += "remfile = %s_remd{}.rem\n" %outputPrefix
            s += "logfile = %s_remd{}.log\n" %outputPrefix
            s += "dcdfile = %s_remd{}.dcd\n" %outputPrefix
            s += "rstfile = %s_remd{}.rst\n" %outputPrefix
            s += "pdbfile = %s_remd{}.pdb\n" %outputPrefix
        else:
            s += "dcdfile = %s.dcd\n" %outputPrefix
            s += "rstfile = %s.rst\n" %outputPrefix
            s += "pdbfile = %s.pdb\n" %outputPrefix

        s += "\n[ENERGY] \n" #-----------------------------------------------------------
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            s += "forcefield = CHARMM \n"
        elif self.forcefield.get() == FORCEFIELD_AAGO:
            s += "forcefield = AAGO  \n"
        elif self.forcefield.get() == FORCEFIELD_CAGO:
            s += "forcefield = CAGO  \n"

        if self.electrostatics.get() == ELECTROSTATICS_CUTOFF :
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
            s += "\n[MINIMIZE]\n" #-----------------------------------------------------------
            s += "method = SD\n"
        else:
            s += "\n[DYNAMICS] \n" #-----------------------------------------------------------
            if self.integrator.get() == INTEGRATOR_VVERLET:
                s += "integrator = VVER  \n"
            elif self.integrator.get() == INTEGRATOR_LEAPFROG:
                s += "integrator = LEAP  \n"
            elif self.integrator.get() == INTEGRATOR_NMMD:
                s += "integrator = NMMD  \n"
            s += "timestep = %f \n" % self.time_step.get()
        s += "nsteps = %i \n" % self.n_steps.get()
        s += "eneout_period = %i \n" % self.eneout_period.get()
        s += "crdout_period = %i \n" % self.crdout_period.get()
        s += "rstout_period = %i \n" % self.n_steps.get()
        s += "nbupdate_period = %i \n" % self.nbupdate_period.get()

        if self.integrator.get() == INTEGRATOR_NMMD:
            s += "\n[NMMD] \n" #-----------------------------------------------------------
            s+= "nm_number = %i \n" % self.nm_number.get()
            s+= "nm_mass = %f \n" % self.nm_mass.get()
            s+= "nm_limit = %f \n" % self.nm_limit.get()
            s+= "elnemo_cutoff = %f \n" % self.elnemo_cutoff.get()
            s+= "elnemo_rtb_block = %i \n" % self.elnemo_rtb_block.get()
            s+= "elnemo_path = %s \n" %  Plugin.getVar("NMA_HOME")
            if self.simulationType.get() == SIMULATION_REMD:
                s+= "nm_prefix = %s_remd{} \n" % outputPrefix
            else:
                s += "nm_prefix = %s \n" % outputPrefix

        s += "\n[CONSTRAINTS] \n" #-----------------------------------------------------------
        if self.rigid_bond.get()        : s += "rigid_bond = YES \n"
        else                            : s += "rigid_bond = NO \n"
        if self.fast_water.get()      :
            s += "fast_water = YES \n"
            s += "water_model = %s \n" %self.water_model.get()
        else                            : s += "fast_water = NO \n"

        s += "\n[BOUNDARY] \n" #-----------------------------------------------------------
        if self.boundary.get() == BOUNDARY_PBC:
            s += "type = PBC \n"
            s += "box_size_x = %f \n" % self.box_size_x.get()
            s += "box_size_y = %f \n" % self.box_size_y.get()
            s += "box_size_z = %f \n" % self.box_size_z.get()
        else :
            s += "type = NOBC \n"

        s += "\n[ENSEMBLE] \n" #-----------------------------------------------------------
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

        if (self.EMfitChoice.get()==EMFIT_VOLUMES)\
                and self.simulationType.get() != SIMULATION_MIN:
            s += "\n[SELECTION] \n" #-----------------------------------------------------------
            s += "group1 = all and not hydrogen\n"

            s += "\n[RESTRAINTS] \n" #-----------------------------------------------------------
            s += "nfunctions = 1 \n"
            s += "function1 = EM \n"
            s += "constant1 = %s \n" % self.constantK.get()
            s += "select_index1 = 1 \n"

            s += "\n[EXPERIMENTS] \n" #-----------------------------------------------------------
            s += "emfit = YES  \n"
            s += "emfit_sigma = %.4f \n" % self.emfit_sigma.get()
            s += "emfit_tolerance = %.6f \n" % self.emfit_tolerance.get()
            s += "emfit_period = 1  \n"
            if self.EMfitChoice.get() == EMFIT_VOLUMES:
                s += "emfit_target = %s.mrc \n" % inputEMprefix

            if self.simulationType.get() == SIMULATION_REMD:
                s += "\n[REMD] \n" #-----------------------------------------------------------
                s += "dimension = 1 \n"
                s += "exchange_period = %i \n" % self.exchange_period.get()
                s += "type1 = RESTRAINT \n"
                s += "nreplica1 = %i \n" % self.nreplica.get()
                s += "rest_function1 = 1 \n"

        with open(inp_file, "w") as f:
            f.write(s)

    # --------------------------- Create output step --------------------------------------------

    def createOutputStep(self):
        """
        Create output set of PDBs
        :return None:
        """
        # CREATE SET OF PDBs
        pdbset = self._createSetOfPDBs("outputPDBs")

        # Add each output PDB to the Set
        for i in range(self.getNumberOfFitting()):

            # Extract the pdb from the DCD file in case of SPDYN
            if self.md_program.get() == PROGRAM_SPDYN:
                lastPDBFromDCD(
                    inputDCD=self.getOutputPrefix(i)+ ".dcd",
                    outputPDB=self.getOutputPrefix(i)+ ".pdb",
                    inputPDB=self.getInputPDBprefix(i)+".pdb")

            outputPrefix =self.getOutputPrefixAll(i)
            for j in outputPrefix:
                pdbset.append(AtomStruct(j + ".pdb"))
        self._defineOutputs(outputPDBs=pdbset)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
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
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------

    def getNumberOfInputPDB(self):
        """
        Get the number of input PDBs
        :return int: number of input PDBs
        """
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
        else: return 0

    def getNumberOfFitting(self):
        """
        Get the number of simulations to perform
        :return int: Number of simulations
        """
        numberOfInputPDB = self.getNumberOfInputPDB()
        numberOfInputEM = self.getNumberOfInputEM()

        # Check input volumes correspond to input PDBs
        if numberOfInputPDB != numberOfInputEM and \
                numberOfInputEM != 1 and numberOfInputPDB != 1:
            raise RuntimeError("Number of input volumes and PDBs must be the same.")
        return np.max([numberOfInputEM, numberOfInputPDB])

    def getInputPDBfn(self):
        """
        Get the input PDB file names
        :return list : list of input PDB file names
        """
        initFn = []
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
        if self.simulationType.get() == SIMULATION_REMD:
            for i in range(self.nreplica.get()):
                outputPrefix.append(self._getExtraPath("%s_output_remd%i" %
                                (str(index + 1).zfill(5), i + 1)))
        else:
            outputPrefix.append(self._getExtraPath("%s_output" % str(index + 1).zfill(5)))
        return outputPrefix

    def getMPIParams(self):
        """
        Get mpi parameters for the simulation
        :return tuple: numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter
        """

        if self.simulationType.get() == SIMULATION_REMD :
            nreplica = self.nreplica.get()
            if nreplica < self.numberOfMpi.get():
                raise RuntimeError("Number of MPI cores should be larger than the number of replicas.")
        else :
            nreplica = 1
        n_fit = self.getNumberOfFitting() * nreplica

        if n_fit <= self.numberOfMpi.get():
            numberOfMpiPerFit   = self.numberOfMpi.get()//self.getNumberOfFitting()
            numberOfLinearFit   = 1
            numberOfParallelFit = self.getNumberOfFitting()
            numberOflastIter    = 0
        else:
            numberOfMpiPerFit   = nreplica
            numberOfLinearFit   = n_fit//self.numberOfMpi.get()
            numberOfParallelFit = self.numberOfMpi.get()//nreplica
            numberOflastIter    = n_fit % self.numberOfMpi.get()

        return numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter

    def getGenesisEnv(self):
        """
        Get environnement for running GENESIS
        :return Environ: environnement
        """
        environ = pwutils.Environ(os.environ)
        environ.set('PATH', os.path.join(Plugin.getVar("GENESIS_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        return environ

    def getGenesisCmd(self, prefix,n_mpi):
        """
        Get GENESIS cmd to run
        :param str prefix: prefix of the simulation
        :param int n_mpi: number of MPI processes
        :return str : GENESIS commadn to run
        """
        cmd=""
        if self.md_program.get() == PROGRAM_ATDYN:
            cmd +=  "atdyn %s " % ("%s_INP" % prefix)
        else:
            cmd += "spdyn %s " % ("%s_INP" % prefix)
        cmd += "  > %s.log" % prefix
        return cmd

    def getRestartFile(self, index=0):
        """
        Get input restart file
        :param int index: Index of the simulation
        :return str: restart file
        """
        rstfile = self.inputRST.get()
        rstList = rstfile.split(" ")
        if len(rstList) >1:
            return rstList[index]
        else:
            return rstList[0]