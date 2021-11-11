# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

K_BOLTZMANN = 1.380649e-23
CARBON_MASS = 12.0107
AVOGADRO_CONST = 6.02214076 * 1e23
KCAL_TO_JOULE = 4186
ATOMIC_MASS_UNIT = 1.66054e-27
ELEMENTARY_CHARGE = 1.602e-19
ANGSTROM_TO_METER = 1e-10
VACUUM_PERMITTIVITY = 8.854*1e-12
WATER_RELATIVE_PERMIT = 80.2

TOPOLOGY_FILE = "data/toppar/top_all36_prot.rtf"
PARAMETER_FILE = "data/toppar/par_all36_prot.prm"

FIT_VAR_LOCAL = "local"
FIT_VAR_GLOBAL = "global"
FIT_VAR_ROTATION = "rotation"
FIT_VAR_SHIFT = "shift"

DEFAULT_FORCEFIELD ={
    "K_BONDS": 305.000,
    "R0_BONDS":  3.796562314087224,
    "THETA0_ANGLES": 102.85,
    "K_ANGLES":  40.000,
    "K_TORSIONS": 3.100,
    "N_TORSIONS": 2,
    "DELTA_TORSIONS": -90.0,
}

DEFAULT_FIT_PARAMS ={
    "biasing_factor" : 1,
    "potential_factor" : 1,

    "potentials" : ["bonds", "angles", "dihedrals"],
    "cutoffpl" : 15.0,
    "cutoffnb" : 10.0,

    "n_step": 10,
    "n_iter": 20,
    "n_warmup": 10,
    "criterion" :False,

    "output_update":10,

    "gradient" : "LS",

    "limit" : 1000,
    "nb_update": 20,

    FIT_VAR_LOCAL+"_dt" : 1e-15,
    "temperature" : 300,
    FIT_VAR_GLOBAL+"_dt" : 10e-15,
    FIT_VAR_GLOBAL+"_sigma" : 1,
    FIT_VAR_ROTATION+"_dt": 0.00001,
    FIT_VAR_ROTATION+"_sigma": 1,
    FIT_VAR_SHIFT+"_dt": 0.00001,
    FIT_VAR_SHIFT+"_sigma": 1,
}
