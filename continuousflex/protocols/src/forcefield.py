import autograd.numpy as npg
from autograd import elementwise_grad

import continuousflex.protocols.src.functions


def get_energy(coord, molstr, molprm, verbose=True):
    """
    Compute the potential energy
    :param coord: the Cartesian coordinates numpy array N*3
    :param molstr: Moleculestructure object
    :param molprm: MoleculeForceFieldPrm object
    :param verbose: verbose level
    :return: Potential energy
    """
    U_bonds = get_energy_bonds(coord, molstr.bonds, molprm)
    U_angles = get_energy_angles(coord, molstr.angles, molprm)
    U_dihedrals = get_energy_dihedrals(coord, molstr.dihedrals, molprm)
    U_total = U_bonds + U_angles + U_dihedrals

    if verbose:
        print("|-- Bonds = " + str(round(U_bonds, 2)))
        print("|-- Angles = " + str(round(U_angles, 2)))
        print("|-- Dihedrals = " + str(round(U_dihedrals, 2)))
        # print("|-- Van der Waals = " + str(round(U_torsions, 2)))
        print("|== TOTAL = " + str(round(U_total, 2)))

    return U_total

def get_autograd(params, mol):
    """
    Compute the gradient of the potential energy by automatic differentiation
    :param params: dictionary with keys are the name of the parameters and values their values
    :param mol: initial Molecule
    :return: gradient for each parameter
    """
    def get_energy_autograd(params, mol):
        """
        Energy function for automatic differentiation
        """
        coord = npg.array(mol.coords)
        if "x" in params:
            coord += params["x"]
        if "q" in params:
            coord += npg.dot(params["q"], mol.modes)
        if "angles" in params:
            coord = npg.dot(continuousflex.protocols.src.functions.generate_euler_matrix(params["angles"]), coord.T).T
        if "shift" in params:
            coord += params["shift"]

        U_bonds = get_energy_bonds(coord, mol.psf.bonds, mol.prm)
        U_angles = get_energy_angles(coord, mol.psf.angles, mol.prm)
        U_dihedrals = get_energy_dihedrals(coord, mol.psf.dihedrals, mol.prm)

        return U_bonds + U_angles + U_dihedrals
    grad = elementwise_grad(get_energy_autograd, 0)
    return grad(params, mol)

def get_energy_bonds(coord, bonds, prm):
    """
    Compute bonds potential
    :param coord: Cartesian coordinates
    :param bonds: bonds index
    :param prm: MoleculeForceFieldPrm
    :return: bonds potential
    """
    r = npg.linalg.norm(coord[bonds[:, 0]] - coord[bonds[:, 1]], axis=1)
    return npg.sum(prm.Kb * npg.square(r - prm.b0))

def get_energy_angles(coord, angles, prm):
    """
    Compute angles potnetial
    :param coord: Cartesian coordinates
    :param angles: angles index
    :param prm: MoleculeForceFieldPrm
    :return: angles potential
    """
    a1 = coord[angles[:, 0]]
    a2 = coord[angles[:, 1]]
    a3 = coord[angles[:, 2]]
    theta = -npg.arccos(npg.sum((a1 - a2) * (a2 - a3), axis=1)
                       / (npg.linalg.norm(a1 - a2, axis=1) * npg.linalg.norm(a2- a3, axis=1))) + npg.pi
    return npg.sum(prm.KTheta * npg.square(theta - (prm.Theta0*npg.pi/180)))

def get_energy_dihedrals(coord, dihedrals, prm):
    """
    Compute dihedrals potnetial
    :param coord: Cartesian coordinates
    :param dihedrals: dihedrals index
    :param prm: MoleculeForceFieldPrm
    :return: dihedrals potential
    """
    u1 = coord[dihedrals[:, 1]] - coord[dihedrals[:, 0]]
    u2 = coord[dihedrals[:, 2]] - coord[dihedrals[:, 1]]
    u3 = coord[dihedrals[:, 3]] - coord[dihedrals[:, 2]]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                           npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(prm.Kchi * (1 + npg.cos(prm.n * (torsions) - (prm.delta*npg.pi/180))))
