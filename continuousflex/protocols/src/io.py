import os

import mrcfile
import numpy as np

from continuousflex.protocols.src.constants import TOPOLOGY_FILE

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

def read_pdb(file):
    """
    Read PDB file
    :param file: PDF file
    :return: coord, atom_type, chain_id, genfile :  the Cartesian coordinates, list of atom types,
                                                    list of chain indexes and generative PDB file
    """
    coords = []
    atom_type=[]
    chain_id =[0]
    with open(file, "r") as f :
        for line in f:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    coords.append([float(l[6]), float(l[7]), float(l[8])])
                    atom_type.append(l[2])
                if l[0] == 'TER':
                    chain_id.append(len(coords))

    if len(chain_id) == 1:
        chain_id.append(len(coords))

    return np.array(coords), np.array(atom_type), chain_id, file

def save_pdb(mol, file):
    """
    Save Molecule to PDB file
    :param mol: Molecule to save
    :param file: PDB file
    """
    genfile = mol.genfile
    print("Saving pdb file ...")
    with open(file, "w") as file:
        with open(genfile, "r") as genfile:
            n=0
            for line in genfile:
                split_line= line.split()

                if len(split_line) > 0:
                    if split_line[0] == 'ATOM':
                        l = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[66:78]]
                        if (mol.coarse_grained and split_line[2] == 'CA') or (not mol.coarse_grained):
                            coord = mol.coords[n]
                            l[0] = l[0].ljust(6)  # atom#6s
                            l[1] = l[1].rjust(5)  # aomnum#5d
                            l[2] = l[2].center(4)  # atomname$#4s
                            l[3] = l[3].ljust(3)  # resname#1s
                            l[4] = l[4].rjust(1)  # Astring
                            l[5] = l[5].rjust(4)  # resnum
                            l[6] = str('%8.3f' % (float(coord[0]))).rjust(8)  # x
                            l[7] = str('%8.3f' % (float(coord[1]))).rjust(8)  # y
                            l[8] = str('%8.3f' % (float(coord[2]))).rjust(8)  # z\
                            l[9] = str('%6.2f' % (float(l[9]))).rjust(6)  # occ
                            l[10] = str('%6.2f' % (float(l[10]))).ljust(6)  # temp
                            l[11] = l[11].rjust(12)  # elname
                            file.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (l[0],l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11]))
                            n+=1

                    elif split_line[0] == 'TER':
                        file.write(line)

                    elif split_line[0] == 'END':
                        # file.write(line)
                        file.write("END\n")
                    # else :
                    #     file.write(line)
    print("Done")

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

def save_mrc(vol, file):
    """
    Save the Volume to an mrc file. The origin of the grid will be (0,0,0)
    :param vol: Volume to save
    :param file: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = vol.data.astype('float32')
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(data.T)
        mrc.voxel_size = vol.voxel_size
        origin = -vol.voxel_size*vol.size/2
        mrc.header['origin']['x'] = origin
        mrc.header['origin']['y'] = origin
        mrc.header['origin']['z'] = origin
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("Done")

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

def create_psf( file):
    """
    Generate PSF file using VMD from a PDB file
    :param file: PDB file to construct PSF file from (the output PSF file will have the same name with only extension changed
    """
    pre, ext = os.path.splitext(file)

    with open("psfgen.tcl", "w") as psfgen :
        psfgen.write("mol load pdb " +file+"\n")
        psfgen.write("set protein [atomselect top protein]\n")
        psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")

        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    set sel [atomselect top \"pfrag $chain\"]\n")
        psfgen.write("    $sel writepdb "+pre+"_tmp${chain}"+ext+"\n")
        psfgen.write("}\n")

        psfgen.write("package require psfgen\n")
        psfgen.write("topology "+TOPOLOGY_FILE+"\n")
        psfgen.write("pdbalias residue HIS HSE\n")
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

    os.system("vmd -dispdev text -e psfgen.tcl")

def read_psf(file):
    """
    Read PSF file
    :param file: PSF file
    :return: dic containing structure data (bonds, angles, dihedrals etc.)
    """
    with open(file) as psf_file:
        dic = {}
        section = {
            "bonds": 2,
            "angles": 3,
            "dihedrals": 4,
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
                    dic["atoms"] = []
                    n = int(split_line[0])
                    n_curr = 0
                    while (n_curr < n):
                        line = psf_file.readline()
                        split_line = line.split()
                        dic["atoms"].append([split_line[0], split_line[4], split_line[5], split_line[6], split_line[7]])
                        n_curr += 1

    return dic

def read_prm(file):
    """
    Rad PRM file (CHARMM parameters file)
    :param file: PRM file
    :return: dic containing parameters data (bonds, angles, dihedrals etc.)
    """
    dic ={
        "bonds" : {},
        "angles" : {},
        "dihedrals" : {}
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

                elif not split_line[0].startswith("!"):
                    if curr == "BONDS":
                        dic["bonds"][split_line[0]+"-"+split_line[1]] = [float(split_line[2]), float(split_line[3])]
                    if curr == "ANGLES":
                        dic["angles"][split_line[0]+"-"+split_line[1]+"-"+split_line[2]] = [float(split_line[3]), float(split_line[4])]
                    if curr == "DIHEDRALS":
                        dic["dihedrals"][split_line[0]+"-"+split_line[1]+"-"+split_line[2]+"-"+split_line[3]] = [float(split_line[4]), int(split_line[5]), float(split_line[6])]
    return dic