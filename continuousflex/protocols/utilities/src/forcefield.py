# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

#import autograd.numpy as npg
#from autograd import elementwise_grad
import numpy as np
import sys
import time

import continuousflex.protocols.utilities.src as src
from continuousflex.protocols.utilities.src.constants import *

def get_energy(coords, forcefield, **kwargs):
    """
    Compute the potential energy
    :param coords: np array N*3 (the Cartesian coordinates(Angstrom))
    :param forcefield: MoleculeForceField object
    :return: Potential energy (kcal * mol-1)
    """
    if "potentials" in kwargs:
        potentials= kwargs["potentials"]
    else:
        potentials = ["bonds", "angles", "dihedrals", "impropers", "urey", "vdw", "elec"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose=False
    U={}
    t= time.time()
    if verbose:
        print("> Computing Potential energy ...")

    # Bonds energy
    if "bonds" in potentials:
        U["bonds"] = get_energy_bonds(coords, forcefield)

    # Angles energy
    if "angles" in potentials:
        U["angles"] = get_energy_angles(coords, forcefield)

    # Dihedrals energy
    if "dihedrals" in potentials:
        U["dihedrals"] = get_energy_dihedrals(coords, forcefield)

    # Impropers energy
    if "impropers" in potentials:
        U["impropers"] = get_energy_impropers(coords, forcefield)

    # Urey Bradley energy
    if "urey" in potentials:
        U["urey"] = get_energy_urey(coords, forcefield)

    # Non bonded energy
    if "vdw" in potentials or "elec" in potentials:
        if "pairlist" in kwargs:
            pairlist = kwargs["pairlist"]
        else:
            if "cutoff" in kwargs:
                cutoff = kwargs["cutoff"]
            else:
                cutoff = 10.0
            pairlist = get_pairlist(coords, excluded_pairs=forcefield.excluded_pairs, cutoff=cutoff)
        invdist = get_invdist(coords, pairlist)

        if "vdw" in potentials:
            U["vdw"] = get_energy_vdw(invdist, pairlist, forcefield)

        if "elec" in potentials:
            U["elec"] = get_energy_elec(invdist, pairlist, forcefield)

    U["total"] = np.sum([U[i] for i in U])
    if verbose:
        for i in U:
            print("\t %-12s %12.2f " % (i, U[i]))
        print("\t %-12s %12.2f " % ("time (s)", time.time()-t))
        print("\t Done \n")

    return U

def get_autograd(params, mol, **kwargs):
    """
    Compute the gradient of the potential energy by automatic differentiation
    :param params: dictionary with keys are the name of the parameters and values their values
    :param mol: initial Molecule
    :return: gradient for each parameter  (kcal * mol-1 * A)
    """
    def update_force(F, F_p):
        for i in F_p:
            if i in F: F[i] += F_p[i]
            else: F[i] = F_p[i]

    def check_force(F, potential):
        for i in F:
            if np.isnan(np.sum(F[i])):
                print("Warning : NaN values encountered in " + potential + "_" + i + " force vector")
                idx_nan = np.where(np.isnan(F[i]))
                F[i][idx_nan] = np.zeros(F[i][idx_nan].shape)

    def limiter(F, potential, **kwargs):
        if ("limit" in kwargs) and (kwargs["limit"] is not None) and "local" in F:
            Fabs = np.linalg.norm(F["local"], axis=1)
            idx = np.where(Fabs > kwargs["limit"])[0]
            if idx.shape[0] > 0:
                print(" Warning : Values beyond limit for force vector " + str(idx.shape[0]) + " " + potential)
                F["local"][idx] = (F["local"][idx].T * kwargs["limit"] / Fabs[idx]).T

    F = {}
    for i in kwargs["potentials"]:
        if i == "bonds" or i == "angles" or i == "dihedrals" or \
            i == "urey" or i == "impropers" :
            force_fct = get_autograd_bonded
        else:
            force_fct = get_autograd_nonbonded
        Fp = force_fct(getattr(src.forcefield, "get_energy_"+i), params, mol, **kwargs)
        check_force(Fp, i)
        if i == "bonds" or i == "vdw":
            limiter(Fp, i, **kwargs)
        update_force(F, Fp)

    return F

def forward_model(params, mol,**kwargs):
    coord = npg.array(mol.coords)
    if FIT_VAR_LOCAL in params:
        coord += params[FIT_VAR_LOCAL]
    if FIT_VAR_GLOBAL in params:
        coord += npg.dot(params[FIT_VAR_GLOBAL], kwargs["normalModeVec"])
    if FIT_VAR_ROTATION in params:
        coord = npg.dot(src.functions.generate_euler_matrix(params[FIT_VAR_ROTATION]), coord.T).T
    if FIT_VAR_SHIFT in params:
        coord += params[FIT_VAR_SHIFT]
    return coord

def get_autograd_bonded(fct, params, mol, **kwargs):
    def get_energy_autograd(params):
        coord = forward_model(params,mol, **kwargs)
        return fct(coord, mol.forcefield)
    return elementwise_grad(get_energy_autograd, 0)(params)

def get_autograd_nonbonded(fct, params, mol, **kwargs):
    def get_energy_autograd(params):
        coord = forward_model(params, mol, **kwargs)
        invdist = get_invdist(coord, kwargs["pairlist"])
        return fct(invdist, kwargs["pairlist"], mol.forcefield)
    return elementwise_grad(get_energy_autograd, 0)(params)

def get_energy_bonds(coord, forcefield):
    """
    Compute bonds potential
    :param coord: Cartesian coordinates (Angstrom)
    :param forcefield: MoleculeForceField
    :return: bonds potential  (kcal * mol-1)
    """
    r = npg.linalg.norm(coord[forcefield.bonds[:, 0]] - coord[forcefield.bonds[:, 1]], axis=1)
    return npg.sum(forcefield.Kb * npg.square(r - forcefield.b0))

def get_energy_angles(coord, forcefield):
    """
    Compute angles potnetial
    :param coord: Cartesian coordinates (Angstrom)
    :param forcefield: MoleculeForceField
    :return: angles potential (kcal * mol-1)
    """
    a1 = coord[forcefield.angles[:, 0]]
    a2 = coord[forcefield.angles[:, 1]]
    a3 = coord[forcefield.angles[:, 2]]
    theta = -npg.arccos(npg.sum((a1 - a2) * (a2 - a3), axis=1)
                       / (npg.linalg.norm(a1 - a2, axis=1) * npg.linalg.norm(a2- a3, axis=1))) + npg.pi
    return npg.sum(forcefield.KTheta * npg.square(theta - (forcefield.Theta0*npg.pi/180)))

def get_energy_dihedrals(coord,forcefield):
    """
    Compute dihedrals potnetial
    :param coord: Cartesian coordinates (Agnstrom)
    :param forcefield: MoleculeForceField
    :return: dihedrals potential  (kcal * mol-1)
    """
    u1 = coord[forcefield.dihedrals[:, 1]] - coord[forcefield.dihedrals[:, 0]]
    u2 = coord[forcefield.dihedrals[:, 2]] - coord[forcefield.dihedrals[:, 1]]
    u3 = coord[forcefield.dihedrals[:, 3]] - coord[forcefield.dihedrals[:, 2]]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                           npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(forcefield.Kchi * (1 + npg.cos(forcefield.n * (torsions) - (forcefield.delta*npg.pi/180))))

def get_energy_impropers(coord,forcefield):
    """
    Compute impropers potnetial
    :param coord: Cartesian coordinates (Agnstrom)
    :param forcefield: MoleculeForceField
    :return: impropers potential  (kcal * mol-1)
    """
    rji = coord[forcefield.impropers[:, 0]] - coord[forcefield.impropers[:, 1]]
    rjk = coord[forcefield.impropers[:, 2]] - coord[forcefield.impropers[:, 1]]
    rkj = -rjk
    rkl = coord[forcefield.impropers[:, 3]] - coord[forcefield.impropers[:, 2]]
    ra = npg.cross(rji, rjk)
    rb = npg.cross(rkj, rkl)
    psi = npg.arccos(npg.sum(ra*rb, axis=1)/ (npg.linalg.norm(ra, axis=1) * npg.linalg.norm(rb, axis=1)))
    return npg.sum(forcefield.Kpsi * (psi - forcefield.psi0*npg.pi/180)**2)

def get_energy_urey(coord, forcefield):
    """
    Compute Urey-Bradley potential
    :param coord: Cartesian coordinates (Angstrom)
    :param forcefield: MoleculeForceField
    :return: urey bradley potential  (kcal * mol-1)
    """
    r = npg.linalg.norm(coord[forcefield.urey[:, 0]] - coord[forcefield.urey[:, 2]], axis=1)
    return npg.sum(forcefield.Kub * npg.square(r - forcefield.S0))

def get_excluded_pairs(forcefield):
    excluded_pairs = {}
    pairs = np.concatenate((forcefield.bonds, forcefield.angles[:,[0,2]]))
    pairs = np.concatenate((pairs, forcefield.dihedral_angles[:,[0,3]]))
    for i in pairs:
        if i[0] in excluded_pairs:
            excluded_pairs[i[0]].append(i[1])
        else:
            excluded_pairs[i[0]]= [i[1]]
        if i[1] in excluded_pairs:
            excluded_pairs[i[1]].append(i[0])
        else:
            excluded_pairs[i[1]]= [i[0]]
    for i in excluded_pairs:
        excluded_pairs[i] = np.array(excluded_pairs[i])
    return excluded_pairs

def get_pairlist(coord, excluded_pairs, cutoff=10.0, verbose=False):
    if verbose : print("Building pairlist ...")
    t=time.time()
    pairlist = []
    # invpairlist= []
    n_atoms=coord.shape[0]
    for i in range(n_atoms):
        dist_idx = np.setdiff1d(np.arange(i + 1, n_atoms), excluded_pairs[i])
        dist = np.linalg.norm(coord[dist_idx] - coord[i], axis=1)
        idx = dist_idx[np.where(dist < cutoff)[0] ]
        # invpairlist.append(idx)
        for j in idx:
            pairlist.append([i, j])
    pl_arr= np.array(pairlist)
    if verbose :
        print("Done ")
        print("\t Size : " + str(sys.getsizeof(pl_arr) / (8 * 1024)) + " kB")
        print("\t Time : " + str(time.time()-t) + " s")
    return pl_arr

def get_invdist(coord, pairlist):
    dist = npg.linalg.norm(coord[pairlist[:, 0]] - coord[pairlist[:, 1]], axis=1)
    return 1/dist

def get_energy_vdw(invdist, pairlist, forcefield):
    Rminij = forcefield.Rmin[pairlist[:, 0]] + forcefield.Rmin[pairlist[:, 1]]
    Epsij = npg.sqrt(forcefield.epsilon[pairlist[:, 0]] * forcefield.epsilon[pairlist[:, 1]])
    invdist6 = (Rminij * invdist) ** 6
    invdist12 = invdist6 ** 2
    return npg.sum(Epsij * (invdist12 - 2 * invdist6)) *2/10

def get_energy_elec(invdist, pairlist, forcefield):
    # Electrostatics
    U = npg.sum( forcefield.charge[pairlist[:, 0]] * forcefield.charge[pairlist[:, 1]] *invdist)
    return U * (ELEMENTARY_CHARGE) ** 2 * AVOGADRO_CONST / \
     (VACUUM_PERMITTIVITY * WATER_RELATIVE_PERMIT * ANGSTROM_TO_METER * KCAL_TO_JOULE)*2
