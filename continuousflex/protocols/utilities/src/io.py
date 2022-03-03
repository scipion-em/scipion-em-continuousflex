# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import os

#import mrcfile
import numpy as np

from continuousflex.protocols.utilities.src.constants import TOPOLOGY_FILE

def read_modes(files):
    """
    Read normal mode vectors list of files
    :param files: list of normal mode files to read
    :return: Normal mode vector Matrix
    """
    A = []
    for i in files:
        A.append(np.loadtxt(i))
    return np.transpose(np.array(A),(1,0,2))

def read_pdb(file, hetatm=False):
    """
    Read PDB file
    :param file: PDF file
    :return: dictionary with pdb data
    """
    atom=[]
    atomNum=[]
    atomName=[]
    resName=[]
    chainName=[]
    resNum=[]
    coords = []
    occ = []
    temp = []
    chainID = []
    elemName=[]
    # print("> Reading pdb file ...")
    with open(file, "r") as f :
        for line in f:
            spl = line.split()
            if len(spl) >0:
                if (spl[0] == 'ATOM') or (hetatm and spl[0][:6] == 'HETATM'):
                    l = [line[:6], line[6:11], line[12:16], line[17:21], line[21:22], line[22:26], line[30:38],
                         line[38:46], line[46:54], line[54:60], line[60:66],line[66:77], line[77:78]]
                    l = [i.strip() for i in l]
                    atom.append(l[0])
                    atomNum.append(l[1])
                    atomName.append(l[2])
                    resName.append(l[3])
                    chainName.append(l[4])
                    resNum.append(l[5])
                    coords.append([float(l[6]), float(l[7]), float(l[8])])
                    occ.append(l[9])
                    temp.append(l[10])
                    chainID.append(l[11])
                    elemName.append(l[12])
    # print("\t Done \n")

    return {
        "atom" : np.array(atom),
        "atomNum" : np.array(atomNum).astype(int),
        "atomName" : np.array(atomName),
        "resName" : np.array(resName),
        "chainName" : np.array(chainName),
        "resNum" : np.array(resNum).astype(int),
        "coords" : np.array(coords).astype(float),
        "occ" : np.array(occ),
        "temp" : np.array(temp),
        "chainID" : np.array(chainID),
        "elemName" : np.array(elemName)
    }

def save_pdb(data, file):
    """
    Save Molecule to PDB file
    :param data: dictionary with pdb data
    :param file: PDB file
    """
    # print("> Saving pdb file ...")
    with open(file, "w") as file:
        past_chainName= data["chainName"][0]
        for i in range(len(data["atom"])):
            if past_chainName != data["chainName"][i]:
                past_chainName = data["chainName"][i]
                file.write("TER\n")

            atom= data["atom"][i].ljust(6)  # atom#6s
            atomNum= str(data["atomNum"][i]).rjust(5)  # aomnum#5d
            atomName= data["atomName"][i].ljust(4)  # atomname$#4s
            resName= data["resName"][i].ljust(4)  # resname#1s
            chainName= data["chainName"][i].rjust(1)  # Astring
            resNum= str(data["resNum"][i]).rjust(4)  # resnum
            coordx= str('%8.3f' % (float(data["coords"][i][0]))).rjust(8)  # x
            coordy= str('%8.3f' % (float(data["coords"][i][1]))).rjust(8)  # y
            coordz= str('%8.3f' % (float(data["coords"][i][2]))).rjust(8)  # z\
            occ= str(data["occ"][i]).rjust(6)  # occ
            temp= str(data["temp"][i]).rjust(6)  # temp
            chainID= str(data["chainID"][i]).rjust(8)  # elname
            elemName= str(data["elemName"][i]).rjust(4)  # elname
            file.write("%s%s  %s%s%s%s    %s%s%s%s%s%s%s\n" % (atom,atomNum, atomName, resName, chainName, resNum,
                                                              coordx, coordy, coordz, occ, temp, chainID, elemName))
        file.write("END\n")
    # print("\t Done \n")

def read_mrc(file):
    """
    Read MRC volume file
    :param file: MRC file
    :return: data, voxel_size : the voxels array of data, the voxel size
    """
    with mrcfile.open(file) as mrc:
        data = np.transpose(mrc.data, (2, 1, 0))
        voxel_size = np.float(mrc.voxel_size['x'])
        return data, voxel_size

def save_mrc(data, file, voxel_size=1, origin=None):
    """
    Save volume data to an mrc file. The origin of the grid will be (0,0,0)
    :param data: volume data to save (array N*N*N)
    :param file: the mrc file name for the output
    :param voxel_size: voxel size
    """
    print("> Saving mrc file ...")
    data = data.astype('float32')
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(data.T)
        mrc.voxel_size = voxel_size
        if origin is None:
            origin = -voxel_size*(data.shape[0])/2   #*0.0 #EDIT
        mrc.header['origin']['x'] = origin
        mrc.header['origin']['y'] = origin
        mrc.header['origin']['z'] = origin
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("\t Done \n")

def read_xmd(file):
    """
    Experimental TODO
    """
    header=[]
    modes = []
    shifts = []
    angles = []
    with open(file, "r") as xmd :
        for line in xmd:
            if not line.startswith("#"):
                split_line = line.split()
                if len(split_line) >0:
                    if split_line[0] == "loop_":
                        while(1):
                            line = xmd.readline()
                            split_line = line.split()
                            if split_line[0].startswith("_"):
                                header.append(split_line[0])
                            else:
                                break
                    if len(split_line) >1:
                        modes.append(np.array(split_line[2:5]).astype(float))
                        shifts.append(np.array(split_line[17:20]).astype(float))
                        angles.append(np.array(split_line[20:23]).astype(float))
    return np.array(modes), np.array(shifts), np.array(angles)

def create_psf( pdb_file,prefix=None, topology_file=None):
    """
    Generate PSF file using VMD from a PDB file
    :param pdb_file: PDB file to construct PSF file from (the output PSF file will have the same name with only extension changed
    :param prefix: output prefix
    :param topology_file: CHARMM TOpology file .rtf
    """
    pre, ext = os.path.splitext(pdb_file)
    if prefix is not None:
        pre = prefix
    if topology_file is None:
        topology_file = TOPOLOGY_FILE


    with open(pre+"_psfgen.tcl", "w") as psfgen :
        psfgen.write("mol load pdb " +pdb_file+"\n")
        psfgen.write("set protein [atomselect top protein]\n")
        psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")

        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    set sel [atomselect top \"pfrag $chain\"]\n")
        psfgen.write("    $sel writepdb "+pre+"_tmp${chain}"+ext+"\n")
        psfgen.write("}\n")

        psfgen.write("package require psfgen\n")
        psfgen.write("topology "+topology_file+"\n")
        psfgen.write("pdbalias residue HIS HSE\n")
        psfgen.write("pdbalias residue MSE MET\n")
        psfgen.write("pdbalias atom ILE CD1 CD\n")

        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    segment U${chain} {pdb "+pre+"_tmp${chain}"+ext+"}\n")
        psfgen.write("    coordpdb "+pre+"_tmp${chain}"+ext+" U${chain}\n")
        psfgen.write("    rm -f "+pre+"_tmp${chain}"+ext+" U${chain}\n")
        psfgen.write("}\n")

        psfgen.write("guesscoord\n")
        psfgen.write("writepdb "+pre+"_PSF.pdb\n")
        psfgen.write("writepsf "+pre+".psf\n")
        psfgen.write("exit\n")

    os.system("vmd -dispdev text -e "+pre+"_psfgen.tcl")

def read_psf(file):
    """
    Read PSF file
    :param file: PSF file
    :return: dic containing structure data (bonds, angles, dihedrals etc.)
    """
    print("> Reading PSF file ... ")
    with open(file) as psf_file:
        dic = {}
        section = {
            "bonds": 2,
            "angles": 3,
            "dihedrals": 4,
            "impropers": 4
        }
        for line in psf_file:
            split_line = line.split()

            if len(split_line) > 2:
                if split_line[1].startswith("!"):
                    for r in section:
                        if r in split_line[2]:
                            dic[r] = []
                            n = int(split_line[0])
                            n_curr = 0
                            while (n_curr < n):
                                line = psf_file.readline()
                                split_line = line.split()
                                for i in range(0, len(split_line), section[r]):
                                    dic[r].append([int(split_line[i + j])-1 for j in range(section[r])])
                                n_curr += len(split_line) // section[r]
                            dic[r] = np.array(dic[r])
                            break

            elif len(split_line) > 1:
                if "NATOM" in split_line[1]:
                    dic["atomNameRes"] = []
                    dic["atomCharge"] = []
                    dic["atomMass"] = []
                    n = int(split_line[0])
                    n_curr = 0
                    while (n_curr < n):
                        line = psf_file.readline()
                        split_line = line.split()
                        dic["atomNameRes"].append(split_line[5])
                        dic["atomCharge"].append(float(split_line[6]))
                        dic["atomMass"].append(float(split_line[7]))
                        n_curr += 1

    print( "\t Done \n")

    return dic

def read_prm(file):
    """
    Rad PRM file (CHARMM parameters file)
    :param file: PRM file
    :return: dic containing parameters data (bonds, angles, dihedrals etc.)
    """
    print("> Reading CHARMM parameter file ...")
    dic ={
        "bonds" : {},
        "angles" : {},
        "dihedrals" : [],
        "impropers": {},
        "nonbonded" : {}
    }
    with open(file) as prm_file :
        curr = ""
        for line in prm_file:
            split_line = line.split()
            if len(split_line)>0 :

                if split_line[0] == "BONDS":
                    curr = "BONDS"
                elif split_line[0] == "ANGLES":
                    curr = "ANGLES"
                elif split_line[0] == "DIHEDRALS":
                    curr = "DIHEDRALS"
                elif split_line[0] == "IMPROPER":
                    curr = "IMPROPER"
                elif split_line[0] == "CMAP":
                    curr = "NONBONDED"
                elif split_line[0] == "NONBONDED":
                    curr = "NONBONDED"
                elif split_line[0] == "HBOND":
                    curr = "HBOND"
                elif split_line[0] == "END":
                    curr = "END"

                elif not split_line[0].startswith("!"):
                    if curr == "BONDS":
                        dic["bonds"][split_line[0]+"-"+split_line[1]] = [float(split_line[2]), float(split_line[3])]
                    if curr == "ANGLES":
                        if len(split_line ) > 5:
                            if not split_line[5].startswith("!") :
                                dic["angles"][split_line[0] + "-" + split_line[1] + "-" + split_line[2]] = [
                                    float(split_line[3]), float(split_line[4]), float(split_line[5]), float(split_line[6])]
                            else:
                                dic["angles"][split_line[0]+"-"+split_line[1]+"-"+split_line[2]] = [float(split_line[3]), float(split_line[4])]
                        else:
                            dic["angles"][split_line[0]+"-"+split_line[1]+"-"+split_line[2]] = [float(split_line[3]), float(split_line[4])]
                    if curr == "DIHEDRALS":
                        dic["dihedrals"].append([split_line[0], split_line[1], split_line[2], split_line[3], float(split_line[4]), int(split_line[5]), float(split_line[6])])
                    if curr == "IMPROPER":
                        dic["impropers"][split_line[0]+"-"+split_line[1]+"-"+split_line[2]+"-"+split_line[3]] = [float(split_line[4]), float(split_line[6])]
                    if curr == "NONBONDED":
                        try :
                            float(split_line[2])
                            float(split_line[3])
                        except ValueError: pass
                        else:
                            dic["nonbonded"][split_line[0]] = [float(split_line[2]), float(split_line[3])]

    print("\t Done \n")
    return dic

def save_mol(file, mol):
    with open(file, "w") as f:
        f.write("#atoms "+str(mol.n_atoms)+"\n")
        for i in range(mol.n_atoms):
            f.write("%12i %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n" %
                (mol.atomNum[i], mol.coords[i][0], mol.coords[i][1], mol.coords[i][2], mol.forcefield.mass[i],
                 mol.forcefield.charge[i], mol.forcefield.Rmin[i], mol.forcefield.epsilon[i]))

        f.write("#bonds " + str(mol.forcefield.n_bonds) + "\n")
        for i in range(mol.forcefield.n_bonds):
            f.write("%12i %12i %12.3f %12.3f\n" %
                    (mol.forcefield.bonds[i][0], mol.forcefield.bonds[i][1],
                     mol.forcefield.Kb[i], mol.forcefield.b0[i]))

        f.write("#angles " + str(mol.forcefield.n_angles) + "\n")
        for i in range(mol.forcefield.n_angles):
            f.write("%12i %12i %12i %12.3f %12.3f\n" %
                    (mol.forcefield.angles[i][0], mol.forcefield.angles[i][1], mol.forcefield.angles[i][2],
                     mol.forcefield.KTheta[i], mol.forcefield.Theta0[i]))

        f.write("#dihedrals " + str(mol.forcefield.n_dihedrals) + "\n")
        for i in range(mol.forcefield.n_dihedrals):
            f.write("%12i %12i %12i %12i %12.3f %12.3f %12.3f\n" %
                    (mol.forcefield.dihedrals[i][0], mol.forcefield.dihedrals[i][1], mol.forcefield.dihedrals[i][2],
                     mol.forcefield.dihedrals[i][2], mol.forcefield.Kchi[i], mol.forcefield.delta[i], mol.forcefield.n[i]))

        f.write("#impropers " + str(mol.forcefield.n_impropers) + "\n")
        for i in range(mol.forcefield.n_impropers):
            f.write("%12i %12i %12i %12i %12.3f %12.3f\n" %
                    (mol.forcefield.impropers[i][0], mol.forcefield.impropers[i][1], mol.forcefield.impropers[i][2],
                     mol.forcefield.impropers[i][2], mol.forcefield.Kpsi[i], mol.forcefield.psi0[i]))

        f.write("#urey " + str(mol.forcefield.n_urey) + "\n")
        for i in range(mol.forcefield.n_urey):
            f.write("%12i %12i %12i %12.3f %12.3f\n" %
                    (mol.forcefield.urey[i][0], mol.forcefield.urey[i][1], mol.forcefield.urey[i][2],
                     mol.forcefield.Kub[i], mol.forcefield.S0[i]))

        f.write("#end\n")

def save_dcd(mol, coords_list, prefix):
    print("> Saving DCD trajectory ...")
    n_frames = len(coords_list)

    # saving PDBs
    mol = mol.copy()
    for i in range(n_frames):
        mol.coords = coords_list[i]
        mol.save_pdb("%s_frame%i.pdb" % (prefix, i))

    # VMD command
    with open(prefix+"_cmd.tcl", "w") as f :
        f.write("mol new %s_frame0.pdb\n" % prefix)
        for i in range(1,n_frames):
            f.write("mol addfile %s_frame%i.pdb\n" % (prefix, i))
        f.write("animate write dcd %s_traj.dcd\n" % prefix)
        f.write('exit')

    # Running VMD
    os.system("vmd -dispdev text -e %s_cmd.tcl" % prefix)

    # Cleaning
    for i in range(n_frames):
        os.system("rm -f %s_frame%i.pdb\n" % (prefix, i))
    os.system("rm -f %s_cmd.tcl" % prefix)
    print("\t Done \n")


def append_dcd(pdb_file, dcd_file, first_frame=False):
    print("> Saving DCD trajectory ...")
    # VMD command
    with open("%s.tcl" % dcd_file, "w") as f :
        if first_frame:
            f.write("mol new\n")
        else:
            f.write("mol load dcd %s\n" % dcd_file)
        f.write("mol addfile %s\n" % pdb_file)
        f.write("animate write dcd %s\n" % dcd_file)
        f.write('exit')

    # Running VMD
    os.system("vmd -dispdev text -e %s.tcl > /dev/null" % dcd_file)

    # Cleaning
    os.system("rm -f %s.tcl" % dcd_file)
    print("\t Done \n")
