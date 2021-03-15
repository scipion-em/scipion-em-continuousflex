import copy
from itertools import permutations
import numpy as np

import continuousflex.protocols.src.forcefield
import continuousflex.protocols.src.functions
import continuousflex.protocols.src.io
import continuousflex.protocols.src.viewers
from continuousflex.protocols.src.constants import *


class Molecule:
    """
    Atomic structure of a molecule
    """

    def __init__(self, coords, modes=None, atom_type=None, chain_id=None, genfile=None, coarse_grained=False):
        """
        Contructor
        :param coords: numpy array N*3 of Cartesian coordinates of N atoms
        :param modes: numpy array of N*M*3 of normal modes vectors of M modes and N atoms
        :param atom_type: list of name of atoms
        :param chain_id: list of number of atoms where the chains of the proteins begins
        :param genfile: PDB file used to generate the class
        :param coarse_grained: boolean of type of Molecula if coarse grained
        """
        self.n_atoms= coords.shape[0]
        self.coords = coords
        self.modes=modes
        self.atom_type = atom_type
        self.genfile=genfile
        self.coarse_grained=coarse_grained
        if chain_id is None:
            self.chain_id = [0,self.n_atoms]
            self.n_chain=1
        else:
            self.chain_id=chain_id
            self.n_chain = len(chain_id) -1

    @classmethod
    def from_file(cls, file):
        """
        Constructor from PDB file
        :param file: PDB file
        :return: Molecule
        """
        coords, atom_type, chain_id, genfile =continuousflex.protocols.src.io.read_pdb(file)
        return cls(coords=coords, atom_type=atom_type, chain_id=chain_id, genfile=genfile)

    @classmethod
    def from_molecule(cls, mol):
        """
        Copy Constructor
        """
        return copy.deepcopy(mol)


    def get_energy(self, verbose=False):
        """
        Compute Potential energy of the object
        :param verbose: verbose level
        :return: the Total Potential energy
        """
        return continuousflex.protocols.src.forcefield.get_energy(coord=self.coords, molstr = self.psf, molprm = self.prm, verbose=verbose)

    def add_modes(self, files, selection=None):
        """
        Add normal modes vectors to the object
        :param files: directory containing the normal modes
        :param n_modes: number of desired normal modes
        """
        if selection is not None:
            files = list(np.array(files)[np.array(selection)-1])
        self.modes = continuousflex.protocols.src.io.read_modes(files)
        if self.modes.shape[0] != self.n_atoms:
            raise RuntimeError("Modes vectors and coordinates do not match : ("+str(self.modes.shape[0])+") != ("+str(self.n_atoms)+")")


    def select_modes(self, selected_modes):
        """
        select specific normal modes vectors
        :param selected_modes: index of selected modes
        """
        self.modes = self.modes[:, selected_modes]

    def get_chain(self, id):
        """
        Return coordinates of a specific chain
        :param id: chain number
        :return: coordinates of the chain
        """
        return self.coords[self.chain_id[id]:self.chain_id[id+1]]

    def select_atoms(self, pattern="CA"):
        """
        Select atoms following a specific pattern, transform the molecule to a coarse-grained model
        :param pattern: CA to select Carbon Alpha atoms
        """
        self.coarse_grained = True
        atom_idx = np.where(self.atom_type == pattern)[0]
        self.coords = self.coords[atom_idx]
        self.n_atoms = self.coords.shape[0]
        self.chain_id= [np.argmin(np.abs(atom_idx - self.chain_id[i])) for i in range(self.n_chain)] + [self.n_atoms]
        if self.modes is not None:
            self.modes = self.modes[atom_idx]

    def center_structure(self):
        """
        Center the structure coordinates around 0
        """
        self.coords -= np.mean(self.coords)

    def show(self):
        """
        Show the structure using matplotlib
        """
        continuousflex.protocols.src.viewers.structures_viewer(self)

    def rotate(self, angles):
        """
        Rotate the molecule
        :param angles: list of 3 Euler angles
        """
        R= continuousflex.protocols.src.functions.generate_euler_matrix(angles)
        self.coords = np.dot(R, self.coords.T).T
        for i in range(self.n_atoms):
            if self.modes is not None :
                self.modes[i] =  np.dot(R , self.modes[i].T).T

    def set_forcefield(self, psf_file=None, prm_file=None):
        """
        Set the force field structure and parameters for the Molecule.
        :param psf_file: .psf file associated to the molecule; If None, default parameters are assigned (CA only)
        :param prm_file: .prm parameter file
        """
        if psf_file is not None:
            self.psf = MoleculeStructure.from_psf_file(psf_file)
            if prm_file is None:
                prm_file = PARAMETER_FILE
            self.prm = MoleculeForcefieldPrm.from_prm_file(self.psf, prm_file=prm_file)
        else:
            self.psf = MoleculeStructure.from_default(self.chain_id)
            self.prm = MoleculeForcefieldPrm.from_default(self.psf)

    def save_pdb(self, file):
        continuousflex.protocols.src.io.save_pdb(self, file)

class MoleculeStructure:
    """
    Structure of the Molecule
    """

    def __init__(self, bonds, angles, dihedrals, atoms):
        """
        Constructor
        :param bonds: numpy array of size Nb*2 of Nb bonds index
        :param angles: numpy array of size Na*3 of Na angles index
        :param dihedrals: numpy array of size Nd*4 of Nd dihedrals index
        :param atoms: TODO
        """
        self.bonds = bonds
        self.angles=angles
        self.dihedrals = dihedrals
        self.atoms=atoms
        self.n_atoms = len(atoms)

    @classmethod
    def from_psf_file(cls, file):
        """
        Constructor from psf file
        :param file: .psf file
        :return: MoleculeStructure
        """
        psf = continuousflex.protocols.src.io.read_psf(file)
        return cls(bonds=psf["bonds"], angles=psf["angles"], dihedrals=psf["dihedrals"], atoms=psf["atoms"])

    @classmethod
    def from_default(cls, chain_id):
        """
        Constructor from default structure (consecutive atoms in a chain are bonded)
        :param chain_id: list of chain index
        :return: MoleculeStructure
        """
        bonds = [[], []]
        angles = [[], [], []]
        dihedrals = [[], [], [], []]

        for i in range(len(chain_id)-1):
            idx = np.arange(chain_id[i], chain_id[i + 1])
            bonds[0] += list(idx[:-1])
            bonds[1] += list(idx[1:])

            angles[0] += list(idx[:-2])
            angles[1] += list(idx[1:-1])
            angles[2] += list(idx[2:])

            dihedrals[0] += list(idx[:-3])
            dihedrals[1] += list(idx[1:-2])
            dihedrals[2] += list(idx[2:-1])
            dihedrals[3] += list(idx[3:])

        bonds = np.array(bonds).T
        angles = np.array(angles).T
        dihedrals = np.array(dihedrals).T

        # TODO : atoms
        return cls(bonds=bonds, angles=angles, dihedrals=dihedrals, atoms=np.zeros(chain_id[-1]))


class MoleculeForcefieldPrm:
    """
    Parameters of the force field of the Molecule
    """

    def __init__(self, Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass):
        """
        Constructor
        :param Kb: bonds K
        :param b0: bonds b0
        :param KTheta: angles K
        :param Theta0:  angles theta0
        :param Kchi: dihedrals K
        :param n: dihedrals n
        :param delta: dihedrals delta
        :param charge: atoms charge
        :param mass: atoms mass
        """
        self.Kb = Kb
        self.b0 = b0
        self.KTheta = KTheta
        self.Theta0 = Theta0
        self.Kchi = Kchi
        self.n = n
        self.delta = delta
        self.charge = charge
        self.mass  = mass

    @classmethod
    def from_prm_file(cls, psf, prm_file):
        """
        Set the force field parameters from a .prm file (CHARMM)
        :param psf: MoleculeStructure
        :param prm_file: .prm parameter file
        :return: MoleculeForcefieldPrm
        """
        charmm_force_field = continuousflex.protocols.src.io.read_prm(prm_file)

        atom_type = []
        for i in range(psf.n_atoms):
            atom_type.append(psf.atoms[i][2])
        atom_type = np.array(atom_type)

        n_bonds = len(psf.bonds)
        n_angles = len(psf.angles)
        n_dihedrals = len(psf.dihedrals)

        Kb = np.zeros(n_bonds)
        b0 = np.zeros(n_bonds)
        KTheta= np.zeros(n_angles)
        Theta0= np.zeros(n_angles)
        Kchi= np.zeros(n_dihedrals)
        n= np.zeros(n_dihedrals)
        delta= np.zeros(n_dihedrals)
        charge= np.array(psf.atoms)[:, 3].astype(float)
        mass= np.array(psf.atoms)[:, 4].astype(float)

        for i in range(n_bonds):
            comb = atom_type[psf.bonds[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                bond = "-".join(perm)
                if bond in charmm_force_field["bonds"]:
                    Kb[i] = charmm_force_field["bonds"][bond][0]
                    b0[i] = charmm_force_field["bonds"][bond][1]
                    found = True
                    break
            if not found:
                print("Err")

        for i in range(n_angles):
            comb = atom_type[psf.angles[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                angle = "-".join(perm)
                if angle in charmm_force_field["angles"]:
                    KTheta[i] = charmm_force_field["angles"][angle][0]
                    Theta0[i] = charmm_force_field["angles"][angle][1]
                    found = True
                    break
            if not found:
                print("Err")

        for i in range(n_dihedrals):
            comb = list(atom_type[psf.dihedrals[i]])
            found = False
            for perm in permutations(comb):
                dihedral = "-".join(perm)
                if dihedral in charmm_force_field["dihedrals"]:
                    Kchi[i] = charmm_force_field["dihedrals"][dihedral][0]
                    n[i] = charmm_force_field["dihedrals"][dihedral][1]
                    delta[i] = charmm_force_field["dihedrals"][dihedral][2]
                    found = True
                    break
            if not found:
                found = False
                for perm in permutations(comb):
                    dihedral = "-".join(['X'] + list(perm[1:3]) + ['X'])
                    if dihedral in charmm_force_field["dihedrals"]:
                        Kchi[i] = charmm_force_field["dihedrals"][dihedral][0]
                        n[i] = charmm_force_field["dihedrals"][dihedral][1]
                        delta[i] = charmm_force_field["dihedrals"][dihedral][2]
                        found = True
                        break
                if not found:
                    print("Err")

        return cls(Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass)

    @classmethod
    def from_default(cls, psf):
        """
        Set the force field parameters from default values (CA only)
        :param psf: MoleculeStructure
        :return: MoleculeForcefieldPrm
        """
        Kb = np.ones(psf.bonds.shape[0]) * K_BONDS
        b0 = np.ones(psf.bonds.shape[0]) * R0_BONDS
        KTheta = np.ones(psf.angles.shape[0]) * K_ANGLES
        Theta0 = np.ones(psf.angles.shape[0]) * THETA0_ANGLES
        Kchi = np.ones(psf.dihedrals.shape[0]) * K_TORSIONS
        n = np.ones(psf.dihedrals.shape[0]) * N_TORSIONS
        delta = np.ones(psf.dihedrals.shape[0]) * DELTA_TORSIONS
        charge = np.zeros(psf.n_atoms)
        mass = np.ones(psf.n_atoms) * CARBON_MASS

        return cls(Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass)
