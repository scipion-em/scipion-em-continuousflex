import numpy as np
import os

class PDBMol:
    def __init__(self, pdb_file):
        """
        Contructor
        :param pdb_file: PDB file
        """
        atom = []
        atomNum = []
        atomName = []
        resName = []
        resAlter = []
        chainName = []
        resNum = []
        coords = []
        occ = []
        temp = []
        chainID = []
        elemName = []
        print("> Reading pdb file %s ..." % pdb_file)
        with open(pdb_file, "r") as f:
            for line in f:
                spl = line.split()
                if len(spl) > 0:
                    if (spl[0] == 'ATOM'):  # or (hetatm and spl[0] == 'HETATM'):
                        l = [line[:6], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:26],
                             line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[72:76], line[76:78]]
                        l = [i.strip() for i in l]
                        atom.append(l[0])
                        atomNum.append(l[1])
                        atomName.append(l[2])
                        resAlter.append(l[3])
                        resName.append(l[4])
                        chainName.append(l[5])
                        resNum.append(l[6])
                        coords.append([float(l[7]), float(l[8]), float(l[9])])
                        occ.append(l[10])
                        temp.append(l[11])
                        chainID.append(l[12])
                        elemName.append(l[13])
        print("\t Done \n")

        atomNum = np.array(atomNum)
        atomNum[np.where(atomNum == "*****")[0]] = "-1"

        self.atom = np.array(atom, dtype='<U6')
        self.n_atoms = len(self.atom)
        self.atomNum = np.array(atomNum).astype(int)
        self.atomName = np.array(atomName, dtype='<U4')
        self.resName = np.array(resName, dtype='<U4')
        self.resAlter = np.array(resAlter, dtype='<U1')
        self.chainName = np.array(chainName, dtype='<U1')
        self.resNum = np.array(resNum).astype(int)
        self.coords = np.array(coords).astype(float)
        self.occ = np.array(occ).astype(float)
        self.temp = np.array(temp).astype(float)
        self.chainID = np.array(chainID, dtype='<U4')
        self.elemName = np.array(elemName, dtype='<U2')

    def save(self, file):
        """
        Save to PDB Format
        :param file: pdb file path
        """
        print("> Saving pdb file %s ..." % file)
        with open(file, "w") as file:
            past_chainName = self.chainName[0]
            past_chainID = self.chainID[0]
            for i in range(len(self.atom)):
                if past_chainName != self.chainName[i] or past_chainID != self.chainID[i]:
                    past_chainName = self.chainName[i]
                    past_chainID = self.chainID[i]
                    file.write("TER\n")

                atom = self.atom[i].ljust(6)  # atom#6s
                if self.atomNum[i] == -1 or self.atomNum[i] >= 100000:
                    atomNum = "99999"  # aomnum#5d
                else:
                    atomNum = str(self.atomNum[i]).rjust(5)  # aomnum#5d
                atomName = self.atomName[i].ljust(3)  # atomname$#4s
                resAlter = self.resAlter[i].ljust(1)  # resAlter#1
                resName = self.resName[i].ljust(4)  # resname#1s
                chainName = self.chainName[i].rjust(1)  # Astring
                resNum = str(self.resNum[i]).rjust(4)  # resnum
                coordx = str('%8.3f' % (float(self.coords[i][0]))).rjust(8)  # x
                coordy = str('%8.3f' % (float(self.coords[i][1]))).rjust(8)  # y
                coordz = str('%8.3f' % (float(self.coords[i][2]))).rjust(8)  # z\
                occ = str('%6.2f' % self.occ[i]).rjust(6)  # occ
                temp = str('%6.2f' % self.temp[i]).rjust(6)  # temp
                chainID = str(self.chainID[i]).ljust(4)  # elname
                elemName = str(self.elemName[i]).rjust(2)  # elname
                file.write("%s%s  %s%s%s%s%s    %s%s%s%s%s      %s%s\n" % (
                atom, atomNum, atomName, resAlter, resName, chainName, resNum,
                coordx, coordy, coordz, occ, temp, chainID, elemName))
            file.write("END\n")
        print("\t Done \n")

    def select_atoms(self, idx):
        self.coords = self.coords[idx]
        self.n_atoms = self.coords.shape[0]
        self.atom = self.atom[idx]
        self.atomNum = self.atomNum[idx]
        self.atomName = self.atomName[idx]
        self.resName = self.resName[idx]
        self.resAlter = self.resAlter[idx]
        self.chainName = self.chainName[idx]
        self.resNum = self.resNum[idx]
        self.elemName = self.elemName[idx]
        self.occ = self.occ[idx]
        self.temp = self.temp[idx]
        self.chainID = self.chainID[idx]

    def get_chain(self, chainName):
        if not isinstance(chainName, list):
            chainName=[chainName]
        chainidx =[]
        for i in chainName:
            idx = np.where(self.chainName == i)[0]
            if len(idx) == 0:
                idx= np.where(self.chainID == i)[0]
            chainidx = chainidx + list(idx)
        return np.array(chainidx)

    def select_chain(self, chainName):
        self.select_atoms(self.get_chain(chainName))

    def remove_alter_atom(self):
        idx = []
        for i in range(self.n_atoms):
            if self.resAlter[i] != "":
                print("!!! Alter residue %s for atom %i"%(self.resName[i], self.atomNum[i]))
                if self.resAlter[i] == "A":
                    idx.append(i)
                    self.resAlter[i]=""
            else:
                idx.append(i)
        self.select_atoms(idx)

    def remove_hydrogens(self):
        idx=[]
        for i in range(self.n_atoms):
            if not self.atomName[i].startswith("H"):
                idx.append(i)
        self.select_atoms(idx)

    def alias_atom(self, atomName, atomNew, resName=None):
        n_alias = 0
        for i in range(self.n_atoms):
            if self.atomName[i] == atomName:
                if resName is not None :
                    if self.resName[i] == resName :
                        self.atomName[i] = atomNew
                        n_alias+=1
                else:
                    self.atomName[i] = atomNew
                    n_alias+=1
        print("%s -> %s : %i lines changed"%(atomName, atomNew, n_alias))

    def alias_res(self, resName, resNew):
        n_alias=0
        for i in range(self.n_atoms):
            if self.resName[i] == resName :
                self.resName[i] = resNew
                n_alias+=1
        print("%s -> %s : %i lines changed"%(resName ,resNew,  n_alias))


    def add_terminal_res(self):
        aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
              "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
        past_chainName = self.chainName[0]
        past_chainID = self.chainID[0]
        for i in range(self.n_atoms-1):
            if past_chainName != self.chainName[i+1] or past_chainID != self.chainID[i+1]:
                if self.resName[i] in aa :
                    print("End of chain %s ; adding terminal residue to %s %i %s"%
                          (past_chainID,self.resName[i],self.resNum[i],self.atomName[i]))
                    resNum = self.resNum[i]
                    j=0
                    while self.resNum[i-j] ==resNum :
                        self.resName[i - j] += "T"
                        j+=1
                else:
                    print("End of chain %s %s %i"% (past_chainID,self.resName[i],self.resNum[i]))
                past_chainName = self.chainName[i+1]
                past_chainID = self.chainID[i+1]


        i = self.n_atoms-1
        if self.resName[i] in aa:
            print("End of chain %s ; adding terminal residue to %s %i %s" % (
            past_chainID, self.resName[i], self.resNum[i], self.atomName[i]))
            resNum = self.resNum[i]
            j = 0
            while self.resNum[i - j] == resNum:
                self.resName[i - j] += "T"
                j += 1
        else:
            print("End of chain %s %s %i" % (past_chainID, self.resName[i], self.resNum[i]))

    def atom_res_reorder(self):
        # Check res order :
        chains = list(set(self.chainID))
        chains.sort()
        new_idx = []
        for c in chains:
            chain_idx = self.get_chain(c)
            resNumlist = list(set(self.resNum[chain_idx]))
            resNumlist.sort()
            for i in range(len(resNumlist)):
                idx = np.where(self.resNum[chain_idx] == resNumlist[i])[0]
                new_idx += list(chain_idx[idx])
        self.select_atoms(np.array(new_idx))

        # reorder atoms and res
        for c in chains:
            chain_idx = self.get_chain(c)
            past_resNum = self.resNum[chain_idx[0]]
            resNum = 1
            for i in range(len(chain_idx)):
                if self.resNum[chain_idx[i]] != past_resNum:
                    past_resNum = self.resNum[chain_idx[i]]
                    resNum += 1
                self.resNum[chain_idx[i]] = resNum
                self.atomNum[chain_idx[i]] = i + 1

    def allatoms2ca(self):
        new_idx = []
        for i in range(self.n_atoms):
            if self.atomName[i] == "CA" or self.atomName[i] == "P":
                new_idx.append(i)
        self.select_atoms(np.array(new_idx))

def matchPDBatoms(mols, ca_only=False):
    print("> Matching PDBs atoms ...")
    n_mols = len(mols)

    if mols[0].chainName[0] in mols[1].chainName:
        chaintype = 0
    elif mols[0].chainID[0] in mols[1].chainID:
        chaintype = 1
    else:
        raise RuntimeError("\t Warning : No matching chains")

    ids = []
    ids_idx = []
    for m in mols :
        id_tmp=[]
        id_idx_tmp=[]
        for i in range(m.n_atoms):
            if (not ca_only) or m.atomName[i] == "CA":
                if chaintype == 0 :
                    id_tmp.append(m.chainName[i] + str(m.resNum[i]) + m.atomName[i])
                else:
                    id_tmp.append(m.chainID[i] + str(m.resNum[i]) + m.atomName[i])
                id_idx_tmp.append(i)
        ids.append(np.array(id_tmp))
        ids_idx.append(np.array(id_idx_tmp))

    idx = []
    for i in range(len(ids[0])):
        idx_line = [ids_idx[0][i]]
        for m in range(1,n_mols):
            idx_tmp = np.where(ids[0][i] == ids[m])[0]
            if len(idx_tmp) == 1:
                idx_line.append(ids_idx[m][idx_tmp[0]])
        if len(idx_line) == n_mols :
            idx.append(idx_line)

    if len(idx)==0:
        print("\t Warning : No matching atoms")
    print("\t Done")

    return np.array(idx)

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

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
            psfgen.write("    set seg ${chain}DNA\n")
            psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
            psfgen.write("    $sel set segid $seg\n")
            psfgen.write("    $sel writepdb tmp.pdb\n")
            psfgen.write("    segment $seg { pdb tmp.pdb }\n")
            psfgen.write("    coordpdb tmp.pdb\n")
            if nucleicChoice == NUCLEIC_DNA:
                psfgen.write("    set resids [lsort -unique [$sel get resid]]\n")
                psfgen.write("    foreach r $resids {\n")
                psfgen.write("        patch DEOX ${chain}DNA:$r\n")
                psfgen.write("    }\n")
            psfgen.write("}\n")
            psfgen.write("regenerate angles dihedrals\n")
            psfgen.write("\n")
        psfgen.write("set protein [atomselect top protein]\n")
        psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    set sel [atomselect top \"pfrag $chain\"]\n")
        psfgen.write("    $sel writepdb tmp.pdb\n")
        psfgen.write("    segment U${chain} {pdb tmp.pdb}\n")
        psfgen.write("    coordpdb tmp.pdb U${chain}\n")
        psfgen.write("}\n")
        psfgen.write("rm -f tmp.pdb\n")
        psfgen.write("\n")
        psfgen.write("guesscoord\n")
        psfgen.write("writepdb %s.pdb\n" % outputPrefix)
        psfgen.write("writepsf %s.psf\n" % outputPrefix)
        psfgen.write("exit\n")

    #Run VMD PSFGEN
    os.system("vmd -dispdev text -e " + fnPSFgen)

    #Clean
    os.system("rm -f " + fnPSFgen)


def generateGROTOP(inputPDB, outputPrefix, forcefield, smog_dir):
    mol = PDBMol(inputPDB)
    mol.remove_alter_atom()
    mol.remove_hydrogens()
    mol.alias_atom("CD", "CD1", "ILE")
    mol.alias_atom("OT1", "O")
    mol.alias_atom("OT2", "OXT")
    mol.alias_res("HSE", "HIS")

    if nucleicChoice == NUCLEIC_RNA:
        mol.alias_res("CYT", "C")
        mol.alias_res("GUA", "G")
        mol.alias_res("ADE", "A")
        mol.alias_res("URA", "U")

    elif nucleicChoice == NUCLEIC_DNA:
        mol.alias_res("CYT", "DC")
        mol.alias_res("GUA", "DG")
        mol.alias_res("ADE", "DA")
        mol.alias_res("THY", "DT")

    mol.alias_atom("O1'", "O1*")
    mol.alias_atom("O2'", "O2*")
    mol.alias_atom("O3'", "O3*")
    mol.alias_atom("O4'", "O4*")
    mol.alias_atom("O5'", "O5*")
    mol.alias_atom("C1'", "C1*")
    mol.alias_atom("C2'", "C2*")
    mol.alias_atom("C3'", "C3*")
    mol.alias_atom("C4'", "C4*")
    mol.alias_atom("C5'", "C5*")
    mol.alias_atom("C5M", "C7")
    mol.add_terminal_res()
    mol.atom_res_reorder()
    mol.save(inputPDB)

    # Run Smog2
    os.system("%s/bin/smog2" % smog_dir+\
               "-i %s -dname %s -%s -limitbondlength -limitcontactlength" %
               (inputPDB, outputPrefix,
                "CA" if forcefield == FORCEFIELD_CAGO else "AA"))

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

    # SELECT CA ATOMS IF CAGO MODEL
    if forcefield == FORCEFIELD_CAGO:
        initPDB = PDBMol(inputPDB)
        initPDB.allatoms2ca()
        initPDB.save(inputPDB)