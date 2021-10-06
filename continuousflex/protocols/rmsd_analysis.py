import os
import numpy as np

def rmsd_analysis(outputPrefix, targetPDB, initPDB ,nsteps,outputPeriod,ca_only):

    def get_mol_conv(mol1,mol2, ca_only=False):
        id1 = []
        id2 = []
        id1_idx = []
        id2_idx = []

        if mol1.chainName[0] in mol2.chainName:
            for i in range(mol1.n_atoms):
                if (not ca_only) or mol1.atomName[i] == "CA":
                    id1.append(mol1.chainName[i] + str(mol1.resNum[i]) + mol1.atomName[i])
                    id1_idx.append(i)
            for i in range(mol2.n_atoms):
                if (not ca_only) or mol2.atomName[i] == "CA":
                    id2.append(mol2.chainName[i] + str(mol2.resNum[i]) + mol2.atomName[i])
                    id2_idx.append(i)
        elif mol1.chainID[0] in mol2.chainID:
            for i in range(mol1.n_atoms):
                if (not ca_only) or mol1.atomName[i] == "CA":
                    id1.append(mol1.chainID[i] + str(mol1.resNum[i]) + mol1.atomName[i])
                    id1_idx.append(i)
            for i in range(mol2.n_atoms):
                if (not ca_only) or mol2.atomName[i] == "CA":
                    id2.append(mol2.chainID[i] + str(mol2.resNum[i]) + mol2.atomName[i])
                    id2_idx.append(i)
        else:
            print("\t Warning : No matching coordinates")
        id1 = np.array(id1)
        id2 = np.array(id2)
        id1_idx = np.array(id1_idx)
        id2_idx = np.array(id2_idx)

        idx = []
        for i in range(len(id1)):
            idx_tmp = np.where(id1[i] == id2)[0]
            if len(idx_tmp) == 1:
                idx.append([id1_idx[i], id2_idx[idx_tmp[0]]])

        if len(idx)==0:
            print("\t Warning : No matching coordinates")

        return np.array(idx)

    def get_RMSD_coords(coords1,coords2):
        return np.sqrt(np.mean(np.square(np.linalg.norm(coords1-coords2, axis=1))))

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
        with open(file, "r") as f :
            for line in f:
                spl = line.split()
                if len(spl) >0:
                    if (spl[0] == 'ATOM') :#or (hetatm and spl[0] == 'HETATM'):
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


    class Molecule:
        """
        Atomic structure of a molecule
        """

        def __init__(self, pdb_file, hetatm=False):
            """
            Contructor
            :param pdb_file: PDB file
            """
            data = read_pdb(pdb_file, hetatm=hetatm)
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
            save_pdb(data = data, file=file)

    with open("%s_dcd2pdb.tcl"%outputPrefix, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s.dcd\n" % (initPDB, outputPrefix)
        s += "set nf [molinfo top get numframes]\n"
        s += "for {set i 0 } {$i < $nf} {incr i} {\n"
        s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" %outputPrefix
        s += "}\n"
        s += "exit\n"
        f.write(s)
    os.system("vmd -dispdev text -e %s_dcd2pdb.tcl > /dev/null"% outputPrefix)

    rmsd = []
    target = Molecule(targetPDB)
    N = (nsteps // outputPeriod)
    mol = Molecule(initPDB)
    idx = get_mol_conv(mol, target, ca_only=ca_only)
    if len(idx) > 0:
        rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
        for i in range(N):
            mol = Molecule(outputPrefix + "tmp" + str(i + 1) + ".pdb")
            rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
    else:
        rmsd = np.zeros(N + 1)
    os.system("rm -f %stmp*" % (outputPrefix))
    np.savetxt(outputPrefix + "_rmsd.txt", rmsd)