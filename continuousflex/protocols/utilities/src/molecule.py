# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import copy
from itertools import permutations
import numpy as np

import continuousflex.protocols.utilities.src as src
from continuousflex.protocols.utilities.src.constants import *


class Molecule:
    """
    Atomic structure of a molecule
    """

    def __init__(self, pdb_file, hetatm=True):
        """
        Contructor
        :param pdb_file: PDB file
        """
        data = src.io.read_pdb(pdb_file, hetatm=hetatm)

        self.coords = data["coords"]
        self.n_atoms = data["coords"].shape[0]
        self.atom = data["atom"]
        self.atomNum = data["atomNum"]
        self.atomName = data["atomName"]
        self.resName = data["resName"]
        self.chainName = data["chainName"]
        self.resNum = data["resNum"]
        self.occ = data["occ"]
        self.temp = data["temp"]
        self.elemName = data["elemName"]
        self.chainID = data["chainID"]
        self.normalModeVec=None
        self.forcefield = None

    def copy(self):
        """
        Copy Molecule Object
        :return: Molecule
        """
        return copy.deepcopy(self)


    def get_energy(self, **kwargs):
        """
        Compute Potential energy of the object
        :return: the Total Potential energy
        """
        return src.forcefield.get_energy(coords=self.coords, forcefield=self.forcefield, **kwargs)

    def center(self):
        """
        Center the coordinates around 0,0,0
        """
        self.coords -= np.mean(self.coords, axis=0)

    def rotate(self, angles):
        """
        Rotate the coordinates
        :param angles: list of 3 Euler angles
        """
        R= src.functions.generate_euler_matrix(angles)
        self.coords = np.dot(R, self.coords.T).T
        if self.normalModeVec is not None:
            for i in range(self.n_atoms):
                self.normalModeVec[i] =  np.dot(R , self.normalModeVec[i].T).T

    def show(self):
        """
        Show the structure using matplotlib
        """
        src.viewers.molecule_viewer(self)

    def set_normalModeVec(self, files, **kwargs):
        """
        Set normal modes vectors to the object
        :param files: directory containing the normal modes
        :param n_modes: number of desired normal modes
        """
        if "selection" in kwargs:
            files = list(np.array(files)[np.array(kwargs["selection"])-1])
        self.normalModeVec = src.io.read_modes(files)
        if self.normalModeVec.shape[0] != self.n_atoms:
            raise RuntimeError("Modes vectors and coordinates do not match : ("+str(self.normalModeVec.shape[0])+") != ("+str(self.n_atoms)+")")

    def set_forcefield(self, **kwargs):
        """
        Set the force field structure and parameters for the Molecule.
        """
        self.forcefield = MoleculeForceField(mol=self,**kwargs)

    def save_pdb(self, file):
        """
        Save to PDB Format
        :param file: pdb file path
        """
        data = {
            "atom" : self.atom,
            "atomNum" : self.atomNum,
            "atomName" : self.atomName,
            "resName" : self.resName,
            "chainName" : self.chainName,
            "resNum" : self.resNum,
            "coords" : self.coords,
            "temp" : self.temp,
            "occ" : self.occ,
            "elemName" : self.elemName,
            "chainID" : self.chainID,
        }
        src.io.save_pdb(data = data, file=file)

    def select_atoms(self, idx):
        self.coords = self.coords[idx]
        self.n_atoms = self.coords.shape[0]
        self.atom = self.atom[idx]
        self.atomNum = self.atomNum[idx]
        self.atomName = self.atomName[idx]
        self.resName = self.resName[idx]
        self.chainName = self.chainName[idx]
        self.resNum = self.resNum[idx]
        self.elemName = self.elemName[idx]
        self.occ = self.occ[idx]
        self.temp = self.temp[idx]
        self.chainID = self.chainID[idx]

        # Normal Modes
        if self.normalModeVec is not None:
            self.normalModeVec = self.normalModeVec[idx]

        # Forcefield
        if self.forcefield is not None:
            self.forcefield.select_atoms(idx)


    def allatoms2carbonalpha(self):
        carbonalpha_idx = np.where(self.atomName == "CA")[0]
        self.select_atoms(carbonalpha_idx)

    def allatoms2backbone(self):
        backbone_idx = []
        for i in range(len(self.atomName)):
            if not self.atomName[i].startswith("H"):
                backbone_idx.append(i)
        backbone_idx = np.array(backbone_idx)
        self.select_atoms(backbone_idx)

    def select_chain(self, chainName):
        if not isinstance(chainName, list):
            chainName=[chainName]
        chainidx =[]
        for i in chainName:
            new_chainName = list(np.where(self.chainName == i)[0])
            chainidx = chainidx + new_chainName
        self.select_atoms(np.array(chainidx))

    def nma_deform(self, q):
        """
        deform molecule using NMA
        :param q: numpy array of M normal modes amplitudes
        :return: deformed Molecule
        """
        new_mol = self.copy()
        new_mol.coords += np.dot(q, self.normalModeVec)
        return new_mol


class MoleculeForceField:
    """
    ForceField & Structure of the Molecule
    """

    def __init__(self, mol, **kwargs):
        if ("psf_file" in kwargs) and ("prm_file" in kwargs) :
            self.set_forcefield_psf(mol, kwargs["psf_file"], kwargs["prm_file"])
        else:
            self.set_forcefield_default(mol)


    def set_forcefield_psf(self, mol, psf_file, prm_file):

        psf = src.io.read_psf(psf_file)
        prm = src.io.read_prm(prm_file)

        print("> Setting up forcefield ...")

        atom_type = np.array(psf["atomNameRes"])

        #####################################
        # Bonds
        #####################################
        self.bonds=psf["bonds"]
        self.n_bonds = len(self.bonds)
        self.Kb = np.zeros(self.n_bonds)
        self.b0 = np.zeros(self.n_bonds)
        for i in range(self.n_bonds):
            comb = atom_type[self.bonds[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                bond = "-".join(perm)
                if bond in prm["bonds"]:
                    self.Kb[i] = prm["bonds"][bond][0]
                    self.b0[i] = prm["bonds"][bond][1]
                    found = True
                    break
            if not found:
                raise RuntimeError("Enable to locale BONDS item in the PRM file")
        print("\t Number of %-12s %12i"%("bonds", self.n_bonds))

        #####################################
        # Angles / Urey Bradley
        #####################################
        self.angles=psf["angles"]
        self.n_angles = len(self.angles)
        self.KTheta = np.zeros(self.n_angles)
        self.Theta0 = np.zeros(self.n_angles)
        self.urey = []
        self.Kub = []
        self.S0 = []
        for i in range(self.n_angles):
            comb = atom_type[self.angles[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                angle = "-".join(perm)
                if angle in prm["angles"]:
                    self.KTheta[i] = prm["angles"][angle][0]
                    self.Theta0[i] = prm["angles"][angle][1]
                    if len(prm["angles"][angle])>2:
                        self.urey.append(self.angles[i])
                        self.Kub.append(prm["angles"][angle][2])
                        self.S0.append(prm["angles"][angle][3])
                    found = True
                    break
            if not found:
                raise RuntimeError("Enable to locale ANGLES item in the PRM file")
        self.urey = np.array(self.urey)
        self.n_urey = len(self.urey)
        self.Kub = np.array(self.Kub)
        self.S0 = np.array(self.S0)

        print("\t Number of %-12s %12i" % ("angles", self.n_angles))
        print("\t Number of %-12s %12i" % ("urey-bradley", self.n_urey))

        #####################################
        # Dihedrals
        #####################################
        self.dihedrals=psf["dihedrals"]
        self.n_dihedrals = len(self.dihedrals)
        self.Kchi  = []
        self.n     = []
        self.delta = []
        n=0
        new_dihe = []
        for i in range(self.n_dihedrals):
            dihe = list(atom_type[self.dihedrals[i]])
            found = False
            for j in prm["dihedrals"]:
                if (j[0] == dihe[0] and j[1] == dihe[1] and j[2] == dihe[2] and j[3] == dihe[3]) or \
                   (j[0] == dihe[3] and j[1] == dihe[2] and j[2] == dihe[1] and j[3] == dihe[0]):
                    n+=1
                    found=True
                    new_dihe.append(self.dihedrals[i])
                    self.Kchi.append(j[4])
                    self.n.append(j[5])
                    self.delta.append(j[6])
            if found ==False:
                for j in prm["dihedrals"]:
                    if (j[0] == "X" and j[1] == dihe[1] and j[2] == dihe[2] and j[3] == "X") or \
                            (j[0] == "X" and j[1] == dihe[2] and j[2] == dihe[1] and j[3] == "X"):
                        n += 1
                        found = True
                        new_dihe.append(self.dihedrals[i])
                        self.Kchi.append(j[4])
                        self.n.append(j[5])
                        self.delta.append(j[6])
            if found == False:
                raise RuntimeError("Enable to locale DIHEDRAL item in the PRM file")
        self.dihedral_angles = self.dihedrals
        self.excluded_pairs = src.forcefield.get_excluded_pairs(forcefield=self)
        self.dihedrals = np.array(new_dihe)
        self.Kchi = np.array(self.Kchi)
        self.n = np.array(self.n)
        self.delta = np.array(self.delta)
        print("\t Number of %-12s %12i" % ("dihedrals", self.n_dihedrals))

        #####################################
        # Impropers
        #####################################
        self.impropers=psf["impropers"]
        self.n_impropers = len(self.impropers)
        self.Kpsi = np.zeros(self.n_impropers)
        self.psi0 = np.zeros(self.n_impropers)
        for i in range(self.n_impropers):
            comb = list(atom_type[self.impropers[i]])
            found = False
            for perm in permutations(comb):
                improper = "-".join(perm)
                if improper in prm["impropers"]:
                    self.Kpsi[i] = prm["impropers"][improper][0]
                    self.psi0[i] = prm["impropers"][improper][1]
                    found = True
                    break
            if not found:
                found = False
                for perm in permutations(comb):
                    improper = "-".join([perm[0], 'X', 'X', perm[3]])
                    if improper in prm["impropers"]:
                        self.Kpsi[i] = prm["impropers"][improper][0]
                        self.psi0[i] = prm["impropers"][improper][1]
                        found = True
                        break
                if not found:
                    raise RuntimeError("Enable to locale IMPROPER item in the PRM file")
        print("\t Number of %-12s %12i" % ("impropers", self.n_impropers))

        #####################################
        # Non-Bonded
        #####################################
        self.charge = np.array(psf["atomCharge"])
        self.mass = np.array(psf["atomMass"])
        self.epsilon = np.zeros(mol.n_atoms)
        self.Rmin = np.zeros(mol.n_atoms)

        for i in range(mol.n_atoms):
            if atom_type[i] in prm["nonbonded"]:
                self.epsilon[i] = prm["nonbonded"][atom_type[i]][0]
                self.Rmin[i] = prm["nonbonded"][atom_type[i]][1]
            else:
                raise RuntimeError("Enable to locale NONBONDED item in the PRM file")

        print("\t Done \n")

    def set_forcefield_default(self, mol):

        chainName = mol.chainName
        chainSet = set(chainName)

        bonds = [[], []]
        angles = [[], [], []]
        dihedrals = [[], [], [], []]

        for i in chainSet:
            idx = np.where(chainName == i)[0]
            bonds[0] += list(idx[:-1])
            bonds[1] += list(idx[1:])

            angles[0] += list(idx[:-2])
            angles[1] += list(idx[1:-1])
            angles[2] += list(idx[2:])

            dihedrals[0] += list(idx[:-3])
            dihedrals[1] += list(idx[1:-2])
            dihedrals[2] += list(idx[2:-1])
            dihedrals[3] += list(idx[3:])

        self.bonds = np.array(bonds).T
        self.angles = np.array(angles).T
        self.dihedrals = np.array(dihedrals).T
        self.dihedral_angles = np.array(dihedrals).T
        self.excluded_pairs = src.forcefield.get_excluded_pairs(forcefield=self)
        self.Kb = np.ones(self.bonds.shape[0])          * DEFAULT_FORCEFIELD["K_BONDS"]
        self.b0 = np.ones(self.bonds.shape[0])          * DEFAULT_FORCEFIELD["R0_BONDS"]
        self.KTheta = np.ones(self.angles.shape[0])     * DEFAULT_FORCEFIELD["K_ANGLES"]
        self.Theta0 = np.ones(self.angles.shape[0])     * DEFAULT_FORCEFIELD["THETA0_ANGLES"]
        self.Kchi = np.ones(self.dihedrals.shape[0])    * DEFAULT_FORCEFIELD["K_TORSIONS"]
        self.n = np.ones(self.dihedrals.shape[0])       * DEFAULT_FORCEFIELD["N_TORSIONS"]
        self.delta = np.ones(self.dihedrals.shape[0])   * DEFAULT_FORCEFIELD["DELTA_TORSIONS"]
        self.charge = np.zeros(len(chainName))
        self.mass = np.ones(len(chainName)) * CARBON_MASS

    def select_atoms(self, idx):
        self.mass = self.mass[idx]
        self.charge = self.charge[idx]

        self.Rmin = self.Rmin[idx]
        self.epsilon = self.epsilon[idx]

        new_bonds, bonds_idx = src.functions.select_idx(param=self.bonds, idx=idx)
        self.bonds = new_bonds
        self.Kb = self.Kb[bonds_idx]
        self.b0 = self.b0[bonds_idx]

        new_angles, angles_idx = src.functions.select_idx(param=self.angles, idx=idx)
        self.angles = new_angles
        self.KTheta = self.KTheta[angles_idx]
        self.Theta0 = self.Theta0[angles_idx]

        new_dihedrals, dihedrals_idx = src.functions.select_idx(param=self.dihedrals, idx=idx)
        self.dihedrals = new_dihedrals
        self.Kchi = self.Kchi[dihedrals_idx]
        self.delta = self.delta[dihedrals_idx]
        self.n = self.n[dihedrals_idx]

        new_impropers, impropers_idx = src.functions.select_idx(param=self.impropers, idx=idx)
        self.impropers = new_impropers
        self.Kpsi = self.Kpsi[impropers_idx]
        self.psi0 = self.psi0[impropers_idx]

        new_urey, urey_idx = src.functions.select_idx(param=self.urey, idx=idx)
        self.urey = new_urey
        self.S0 = self.S0[urey_idx]
        self.Kub = self.Kub[urey_idx]

        self.excluded_pairs = src.forcefield.get_excluded_pairs(forcefield=self)
