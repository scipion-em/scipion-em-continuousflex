import numpy as np
import os
from pyworkflow.utils import runCommand, buildRunCommand
from xmippLib import SymList
import pwem.emlib.metadata as md
import sys
from subprocess import Popen
import re


from continuousflex.protocols.utilities.pdb_handler import ContinuousFlexPDBHandler


EMFIT_NONE = 0
EMFIT_VOLUMES = 1
EMFIT_IMAGES = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

SIMULATION_MIN = 0
SIMULATION_MD = 1
SIMULATION_NMMD = 2
SIMULATION_REMD = 3
SIMULATION_RENMMD = 4

PROGRAM_ATDYN = 0
PROGRAM_SPDYN= 1

INTEGRATOR_VVERLET = 0
INTEGRATOR_LEAPFROG = 1

IMPLICIT_SOLVENT_GBSA = 0
IMPLICIT_SOLVENT_NONE = 1

TPCONTROL_NONE = 0
TPCONTROL_LANGEVIN = 1
TPCONTROL_BERENDSEN = 2
TPCONTROL_BUSSI = 3

ENSEMBLE_NVT = 0
ENSEMBLE_NVE = 1
ENSEMBLE_NPT = 2

BOUNDARY_NOBC = 0
BOUNDARY_PBC = 1

ELECTROSTATICS_PME = 0
ELECTROSTATICS_CUTOFF = 1

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

RB_PROJMATCH = 0
RB_WAVELET = 1

def generatePSF(inputPDB, inputTopo, outputPrefix, nucleicChoice):
    fnPSFgen = outputPrefix+"psfgen.tcl"
    with open(fnPSFgen, "w") as psfgen:
        psfgen.write("mol load pdb %s\n" % inputPDB)
        psfgen.write("\n")
        psfgen.write("package require psfgen\n")
        psfgen.write("topology %s\n" % inputTopo)
        psfgen.write("pdbalias residue HIS HSE\n")
        psfgen.write("pdbalias residue MSE MET\n")
        psfgen.write("pdbalias atom ILE CD1 CD\n")
        if nucleicChoice == NUCLEIC_RNA:
            psfgen.write("pdbalias residue A ADE\n")
            psfgen.write("pdbalias residue G GUA\n")
            psfgen.write("pdbalias residue C CYT\n")
            psfgen.write("pdbalias residue U URA\n")
        elif nucleicChoice == NUCLEIC_DNA:
            psfgen.write("pdbalias residue DA ADE\n")
            psfgen.write("pdbalias residue DG GUA\n")
            psfgen.write("pdbalias residue DC CYT\n")
            psfgen.write("pdbalias residue DT THY\n")
        psfgen.write("\n")
        if nucleicChoice == NUCLEIC_RNA or nucleicChoice == NUCLEIC_DNA:
            psfgen.write("set nucleic [atomselect top nucleic]\n")
            psfgen.write("set chains [lsort -unique [$nucleic get chain]] ;\n")
            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
            psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
            psfgen.write("    segment N${chain} { pdb %s_tmp.pdb }\n" % outputPrefix)
            psfgen.write("    coordpdb %s_tmp.pdb N${chain}\n" % outputPrefix)
            if nucleicChoice == NUCLEIC_DNA:
                psfgen.write("    set resids [lsort -unique [$sel get resid]]\n")
                psfgen.write("    foreach r $resids {\n")
                psfgen.write("        patch DEOX N${chain}:$r\n")
                psfgen.write("    }\n")
            psfgen.write("}\n")
            if nucleicChoice == NUCLEIC_DNA:
                psfgen.write("regenerate angles dihedrals\n")
            psfgen.write("\n")
        psfgen.write("set protein [atomselect top protein]\n")
        psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    set sel [atomselect top \"protein and pfrag $chain\"]\n")
        psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
        psfgen.write("    segment P${chain} {pdb %s_tmp.pdb}\n" % outputPrefix)
        psfgen.write("    coordpdb %s_tmp.pdb P${chain}\n" % outputPrefix)
        psfgen.write("}\n")
        psfgen.write("rm -f %s_tmp.pdb\n" % outputPrefix)
        psfgen.write("\n")
        psfgen.write("guesscoord\n")
        psfgen.write("writepdb %s.pdb\n" % outputPrefix)
        psfgen.write("writepsf %s.psf\n" % outputPrefix)
        psfgen.write("exit\n")

    #Run VMD PSFGEN
    runCommand("vmd -dispdev text -e %s > %s.log " %(fnPSFgen,outputPrefix))

    # Check PDB
    outMol = ContinuousFlexPDBHandler(outputPrefix+".pdb")
    if outMol.n_atoms == 0:
        raise RuntimeError("VMD psfgen failed, check %s.log for details"%outputPrefix)

    #Clean
    os.system("rm -f " + fnPSFgen)


def generateGROTOP(inputPDB, outputPrefix, forcefield, smog_dir, nucleicChoice):
    mol = ContinuousFlexPDBHandler(inputPDB)
    # mol.remove_alter_atom()
    mol.remove_hydrogens()
    mol.check_res_order()

    moltmp = mol.copy()

    moltmp.alias_atom("CD", "CD1", "ILE")
    moltmp.alias_atom("OT1", "O")
    moltmp.alias_atom("OT2", "OXT")
    moltmp.alias_res("HSE", "HIS")

    if nucleicChoice == NUCLEIC_RNA:
        moltmp.alias_res("CYT", "C")
        moltmp.alias_res("GUA", "G")
        moltmp.alias_res("ADE", "A")
        moltmp.alias_res("URA", "U")

    elif nucleicChoice == NUCLEIC_DNA:
        moltmp.alias_res("CYT", "DC")
        moltmp.alias_res("GUA", "DG")
        moltmp.alias_res("ADE", "DA")
        moltmp.alias_res("THY", "DT")

    moltmp.alias_atom("O1'", "O1*")
    moltmp.alias_atom("O2'", "O2*")
    moltmp.alias_atom("O3'", "O3*")
    moltmp.alias_atom("O4'", "O4*")
    moltmp.alias_atom("O5'", "O5*")
    moltmp.alias_atom("C1'", "C1*")
    moltmp.alias_atom("C2'", "C2*")
    moltmp.alias_atom("C3'", "C3*")
    moltmp.alias_atom("C4'", "C4*")
    moltmp.alias_atom("C5'", "C5*")
    moltmp.alias_atom("C5M", "C7")
    moltmp.add_terminal_res()
    moltmp.atom_res_reorder()
    moltmp.write_pdb(inputPDB)

    # Run Smog2
    runCommand("%s/bin/smog2" % smog_dir+\
               " -i %s -dname %s -%s -limitbondlength -limitcontactlength > %s.log" %
               (inputPDB, outputPrefix,
                "CA" if forcefield == FORCEFIELD_CAGO else "AA", outputPrefix))


    if forcefield == FORCEFIELD_CAGO:
        mol.select_atoms(mol.allatoms2ca())
    mol.write_pdb(outputPrefix+".pdb")

    # ADD CHARGE TO TOP FILE
    grotopFile = outputPrefix + ".top"
    with open(grotopFile, 'r') as f1:
        with open(grotopFile + ".tmp", 'w') as f2:
            atom_scope = False
            write_line = False
            for line in f1:
                if "[" in line and "]" in line:
                    if "atoms" in line:
                        atom_scope = True
                if atom_scope:
                    if "[" in line and "]" in line:
                        if not "atoms" in line:
                            atom_scope = False
                            write_line = False
                    elif not ";" in line and not (not line or line.isspace()):
                        write_line = True
                    else:
                        write_line = False
                if write_line:
                    f2.write("%s\t0.0\n" % line[:-1])
                else:
                    f2.write(line)
    os.system("cp %s.tmp %s" % (grotopFile, grotopFile))
    os.system("rm -f %s.tmp" % grotopFile)

def save_dcd(mol, coords_list, prefix):
    print("> Saving DCD trajectory ...")
    n_frames = len(coords_list)

    # saving PDBs
    mol = mol.copy()
    for i in range(n_frames):
        mol.coords = coords_list[i]
        mol.write_pdb("%s_frame%i.pdb" % (prefix, i))

    # VMD command
    with open(prefix+"_cmd.tcl", "w") as f :
        f.write("mol new %s_frame0.pdb\n" % prefix)
        for i in range(1,n_frames):
            f.write("mol addfile %s_frame%i.pdb\n" % (prefix, i))
        f.write("animate write dcd %s.dcd\n" % prefix)
        f.write('exit\n')

    # Running VMD
    runCommand("vmd -dispdev text -e %s_cmd.tcl" % prefix)

    # Cleaning
    for i in range(n_frames):
        runCommand("rm -f %s_frame%i.pdb\n" % (prefix, i))
    runCommand("rm -f %s_cmd.tcl" % prefix)
    print("\t Done \n")

def readLogFile(log_file):
    with open(log_file,"r") as file:
        header = None
        dic = {}
        for line in file:
            if line.startswith("INFO:"):
                if header is None:
                    header = line.split()
                    for i in range(1,len(header)):
                        dic[header[i]] = []
                else:
                    splitline = line.split()
                    if len(splitline) == len(header):
                        for i in range(1,len(header)):
                            try :
                                dic[header[i]].append(float(splitline[i]))
                            except ValueError:
                                pass

    return dic

def lastPDBFromDCD(inputPDB,inputDCD,  outputPDB):

    # EXTRACT PDB from dcd file
    with open("%s_tmp_dcd2pdb.tcl" % outputPDB, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s\n" % (inputPDB, inputDCD)
        s += "set nf [molinfo top get numframes]\n"
        s += "[atomselect top all frame [expr $nf - 1]] writepdb %s\n" % outputPDB
        s += "exit\n"
        f.write(s)
    runCommand("vmd -dispdev text -e %s_tmp_dcd2pdb.tcl" % outputPDB)

    # CLEAN TMP FILES
    runCommand("rm -f %s_tmp_dcd2pdb.tcl" % (outputPDB))

def pdb2vol(inputPDB, outputVol, sampling_rate, image_size):
    """
    Create a density volume from a pdb
    :param str inputPDB: input pdb file name
    :param str outputVol: output vol file name
    :param float sampling_rate: Sampling rate
    :param int image_size: Size of the output volume
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_volume_from_pdb"
    args = "-i %s  -o %s --sampling %f --size %i %i %i --centerPDB"%\
           (inputPDB, outputVol,sampling_rate,image_size,image_size,image_size)
    return cmd+ " "+ args

def projectVol(inputVol, outputProj, expImage, sampling_rate=5.0, angular_distance=-1, compute_neighbors=True):
    """
    Create a set of projections from an input volume
    :param str inputVol: Input volume file name
    :param str outputProj: Output set of proj file name
    :param str expImage: Experimental image to project in the neighborhood
    :param float sampling_rate: Samplign rate
    :param float angular_distance: Do not search a distance larger than...
    :param bool compute_neighbors: Compute projection nearby the experimental image
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_project_library"
    args = "-i %s.vol -o %s.stk --sampling_rate %f " % (inputVol, outputProj, sampling_rate)
    if compute_neighbors :
        args +="--compute_neighbors --angular_distance %f " % angular_distance
        args += "--experimental_images %s "%expImage
        if angular_distance != -1 :
            args += "--near_exp_data"
    return cmd+ " "+ args

def projectMatch(inputImage, inputProj, outputMeta):
    """
    Projection matching of an input experimental image with a set of projections
    :param str inputImage: File name of the input experimental image
    :param str inputProj: File name of the input set of projections
    :param str outputMeta: File name of the output Xmipp metadata file with the angles of the matching
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_projection_matching "
    args= "-i %s -o %s --ref %s.stk "%(inputImage, outputMeta, inputProj)
    args +="--search5d_shift 10.0 --search5d_step 1.0"
    return cmd + " "+ args

def waveletAssignement(inputImage, inputProj, outputMeta):
    """
    Make a discrete angular assignment of angles from a set of projections
    :param str inputImage: File name of input experimental image
    :param str inputProj: File name of the input set of projections
    :param str outputMeta: File name of the output Xmipp metadata file with the angles assigned
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_discrete_assign "
    args= "-i %s -o %s --ref %s.doc "%(inputImage, outputMeta, inputProj)
    args +="--psi_step 5.0 --max_shift_change 10.0 --search5D"
    return cmd + " "+ args

def continuousAssign(inputMeta, inputVol, outputMeta):
    """
    Make a continuous angular assignment of angles from a volume
    :param str inputMeta: File name of input Xmipp metadata file with angles to assign
    :param str inputVol: File name of the input Volume
    :param str outputMeta: File name of the output Xmipp metadata file with the angles
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_continuous_assign "
    args= "-i %s -o %s --ref %s.vol "%(inputMeta, outputMeta, inputVol)
    return cmd + " "+ args

def flipAngles(inputMeta, outputMeta):
    """
    Flip angles from Xmipp representation to Euler angles
    :param str inputMeta : File name of input Xmipp metadata file containing the angles to flip
    :param str outputMeta: file name of the output Xmipp matadata file
    :return None:
    """
    Md1 = md.MetaData(inputMeta)
    flip = Md1.getValue(md.MDL_FLIP, 1)
    tilt1 = Md1.getValue(md.MDL_ANGLE_TILT, 1)
    psi1 = Md1.getValue(md.MDL_ANGLE_PSI, 1)
    x1 = Md1.getValue(md.MDL_SHIFT_X, 1)
    if flip:
        Md1.setValue(md.MDL_SHIFT_X, -x1, 1)
        Md1.setValue(md.MDL_ANGLE_TILT, tilt1 + 180, 1)
        Md1.setValue(md.MDL_ANGLE_PSI, -psi1, 1)
    Md1.write(outputMeta)

# def getAngularDist(md1, md2, idx1=1, idx2=1):
#     rot1        = md1.getValue(md.MDL_ANGLE_ROT, int(idx1))
#     tilt1       = md1.getValue(md.MDL_ANGLE_TILT, int(idx1))
#     psi1        = md1.getValue(md.MDL_ANGLE_PSI, int(idx1))
#     rot2        = md2.getValue(md.MDL_ANGLE_ROT, int(idx2))
#     tilt2       = md2.getValue(md.MDL_ANGLE_TILT, int(idx2))
#     psi2        = md2.getValue(md.MDL_ANGLE_PSI, int(idx2))
#
#     return  SymList.computeDistanceAngles(SymList(), rot1, tilt1, psi1, rot2, tilt2, psi2, False, True, False)
#
#
# def getShiftDist(md1, md2, idx1=1, idx2=1):
#     shiftx1 = md1.getValue(md.MDL_SHIFT_X, int(idx1))
#     shifty1 = md1.getValue(md.MDL_SHIFT_Y, int(idx1))
#     shiftx2 = md2.getValue(md.MDL_SHIFT_X, int(idx2))
#     shifty2 = md2.getValue(md.MDL_SHIFT_Y, int(idx2))
#     return np.linalg.norm(np.array([shiftx1, shifty1, 0.0]) - np.array([shiftx2, shifty2, 0.0]))

def getAngularShiftDist(angle1MetaFile, angle2MetaData, angle2Idx, tmpPrefix, symmetry):

    mdImgTmp = md.MetaData()
    mdImgTmp.addObject()
    mdImgTmp.setValue(md.MDL_ANGLE_ROT, angle2MetaData.getValue(md.MDL_ANGLE_ROT, angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_ANGLE_TILT,angle2MetaData.getValue(md.MDL_ANGLE_TILT,angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_ANGLE_PSI, angle2MetaData.getValue(md.MDL_ANGLE_PSI, angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_SHIFT_X,   angle2MetaData.getValue(md.MDL_SHIFT_X,   angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_SHIFT_Y,   angle2MetaData.getValue(md.MDL_SHIFT_Y,   angle2Idx), 1)
    mdImgTmp.write(tmpPrefix + ".xmd")

    cmd = "xmipp_angular_distance --ang1 %s --ang2 %s.xmd --oroot %sDist --sym %s --check_mirrors > %s.log" % \
          (angle1MetaFile, tmpPrefix, tmpPrefix, symmetry, tmpPrefix)
    runCommand(cmd)
    with open(tmpPrefix + ".log", "r") as f:
        for line in f:
            if "angular" in line:
                angDist = float(re.findall("\d+\.\d+", line)[0])
            if "shift" in line:
                shftDist = float(re.findall("\d+\.\d+", line)[0])
    return angDist, shftDist


def dcd2numpyArr(filename):
    print("> Reading dcd file %s"%filename)
    with open(filename, 'rb') as f:

        # Header
        # ---------------- INIT

        start_size = int.from_bytes((f.read(4)), "little")
        crd_type = f.read(4).decode('ascii')
        nframe = int.from_bytes((f.read(4)), "little")
        start_frame = int.from_bytes((f.read(4)), "little")
        len_frame = int.from_bytes((f.read(4)), "little")
        len_total = int.from_bytes((f.read(4)), "little")
        for i in range(5):
            f.read(4)
        time_step = np.frombuffer(f.read(4), dtype=np.float32)
        for i in range(9):
            f.read(4)
        charmm_version = int.from_bytes((f.read(4)), "little")

        end_size = int.from_bytes((f.read(4)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- TITLE

        start_size = int.from_bytes((f.read(4)), "little")
        ntitle = int.from_bytes((f.read(4)), "little")
        title = f.read(80 * ntitle).decode('ascii')
        end_size = int.from_bytes((f.read(4)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- NATOM

        start_size = int.from_bytes((f.read(4)), "little")
        natom = int.from_bytes((f.read(4)), "little")
        end_size = int.from_bytes((f.read(4)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ----------------- DCD COORD
        dcd_list = []
        for i in range(nframe):
            coordarr = np.zeros((natom, 3))
            for j in range(3):

                start_size = int.from_bytes((f.read(4)), "little")
                while (start_size != 4 * natom):
                    # print("\n-- UNKNOWN %s -- " % start_size)

                    f.read(start_size)
                    end_size = int.from_bytes((f.read(4)), "little")
                    if end_size != start_size:
                        raise RuntimeError("Can not read dcd file")
                    start_size = int.from_bytes((f.read(4)), "little")

                bin_arr = f.read(4 * natom)
                if len(bin_arr) == 4 * natom:
                    coordarr[:, j] = np.frombuffer(bin_arr, dtype=np.float32)
                else:
                    break
                end_size = int.from_bytes((f.read(4)), "little")
                if end_size != start_size:
                    if i>1:
                        break
                    else:
                        pass
                        # raise RuntimeError("Can not read dcd file %i %i " % (start_size, end_size))

            dcd_list.append(coordarr)

    print("\t Done \n")

    return np.array(dcd_list)

def existsCommand(name):
    from shutil import which
    return which(name) is not None