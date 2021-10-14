import numpy as np

def smog_pdb(inputPDB, outputPDB):
    def read_pdb(file):
        """
        Read PDB file
        :param file: PDF file
        :return: dictionary with pdb data
        """
        atom=[]
        atomNum=[]
        atomName=[]
        resName=[]
        resAlter=[]
        chainName=[]
        resNum=[]
        coords = []
        occ = []
        temp = []
        chainID = []
        elemName=[]
        print("> Reading pdb file %s ..."%file)
        with open(file, "r") as f :
            for line in f:
                spl = line.split()
                if len(spl) >0:
                    if (spl[0] == 'ATOM') :
                        l = [line[:6], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:26], line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66],line[72:76], line[76:78]]
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

        return {
            "atom" : np.array(atom, dtype='<U6'),
            "atomNum" : np.array(atomNum).astype(int),
            "atomName" : np.array(atomName, dtype='<U4'),
            "resName" : np.array(resName, dtype='<U4'),
            "resAlter" : np.array(resAlter, dtype='<U1'),
            "chainName" : np.array(chainName, dtype='<U1'),
            "resNum" : np.array(resNum).astype(int),
            "coords" : np.array(coords).astype(float),
            "occ" : np.array(occ).astype(float),
            "temp" : np.array(temp).astype(float),
            "chainID" : np.array(chainID, dtype='<U4'),
            "elemName" : np.array(elemName, dtype='<U2')
        }

    def save_pdb(data, file):
        """
        Save Molecule to PDB file
        :param data: dictionary with pdb data
        :param file: PDB file
        """
        print("> Saving pdb file %s ..."%file)
        with open(file, "w") as file:
            past_chainName= data["chainName"][0]
            past_chainID = data["chainID"][0]
            for i in range(len(data["atom"])):
                if past_chainName != data["chainName"][i] or past_chainID != data["chainID"][i] :
                    past_chainName = data["chainName"][i]
                    past_chainID = data["chainID"][i]
                    file.write("TER\n")

                atom= data["atom"][i].ljust(6)  # atom#6s
                if data["atomNum"][i] == -1 or data["atomNum"][i] >=100000:
                    atomNum= "99999"  # aomnum#5d
                else:
                    atomNum= str(data["atomNum"][i]).rjust(5)  # aomnum#5d
                atomName= data["atomName"][i].ljust(4)  # atomname$#4s
                resAlter= data["resAlter"][i]  # resAlter#1
                resName= data["resName"][i].ljust(4)  # resname#1s
                chainName= data["chainName"][i].rjust(1)  # Astring
                resNum= str(data["resNum"][i]).rjust(4)  # resnum
                coordx= str('%8.3f' % (float(data["coords"][i][0]))).rjust(8)  # x
                coordy= str('%8.3f' % (float(data["coords"][i][1]))).rjust(8)  # y
                coordz= str('%8.3f' % (float(data["coords"][i][2]))).rjust(8)  # z\
                occ= str('%6.2f'%data["occ"][i]).rjust(6)  # occ
                temp= str('%6.2f'%data["temp"][i]).rjust(6)  # temp
                chainID= str(data["chainID"][i]).ljust(4)  # elname
                elemName= str(data["elemName"][i]).rjust(2)  # elname
                file.write("%s%s  %s%s%s%s%s    %s%s%s%s%s      %s%s\n" % (atom,atomNum, atomName, resAlter, resName, chainName, resNum,
                                                                  coordx, coordy, coordz, occ, temp, chainID, elemName))
            file.write("END\n")
        print("\t Done \n")

    class Molecule:
        """
        Atomic structure of a molecule
        """

        def __init__(self, pdb_file):
            """
            Contructor
            :param pdb_file: PDB file
            """
            data = read_pdb(pdb_file)

            self.coords = data["coords"]
            self.n_atoms = data["coords"].shape[0]
            self.atom = data["atom"]
            self.atomNum = data["atomNum"]
            self.atomName = data["atomName"]
            self.resName = data["resName"]
            self.resAlter = data["resAlter"]
            self.chainName = data["chainName"]
            self.resNum = data["resNum"]
            self.occ = data["occ"]
            self.temp = data["temp"]
            self.elemName = data["elemName"]
            self.chainID = data["chainID"]
            self.normalModeVec=None
            self.forcefield = None

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
                "resAlter" : self.resAlter,
                "chainName" : self.chainName,
                "resNum" : self.resNum,
                "coords" : self.coords,
                "temp" : self.temp,
                "occ" : self.occ,
                "elemName" : self.elemName,
                "chainID" : self.chainID,
            }
            save_pdb(data = data, file=file)

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

            # Normal Modes
            if self.normalModeVec is not None:
                self.normalModeVec = self.normalModeVec[idx]

            # Forcefield
            if self.forcefield is not None:
                self.forcefield.select_atoms(idx)

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

    def remove_alter_atom(mol):
        idx = []
        for i in range(mol.n_atoms):
            if mol.resAlter[i] != "":
                print("!!! Alter residue %s for atom %i"%(mol.resName[i], mol.atomNum[i]))
                if mol.resAlter[i] == "A":
                    idx.append(i)
                    mol.resAlter[i]=""
            else:
                idx.append(i)
        mol.select_atoms(idx)

    def remove_hydrogens(mol):
        idx=[]
        for i in range(mol.n_atoms):
            if not mol.atomName[i].startswith("H"):
                idx.append(i)
        mol.select_atoms(idx)

    def alias_atom(mol, atomName, atomNew, resName=None):
        n_alias = 0
        for i in range(mol.n_atoms):
            if mol.atomName[i] == atomName:
                if resName is not None :
                    if mol.resName[i] == resName :
                        mol.atomName[i] = atomNew
                        n_alias+=1
                else:
                    mol.atomName[i] = atomNew
                    n_alias+=1
        print("%i lines changed"%n_alias)

    def alias_res(mol, resName, resNew):
        n_alias=0
        for i in range(mol.n_atoms):
            if mol.resName[i] == resName :
                mol.resName[i] = resNew
                n_alias+=1
        print("%i lines changed"%n_alias)


    def add_terminal_res(mol):
        aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
              "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
        past_chainName = mol.chainName[0]
        past_chainID = mol.chainID[0]
        for i in range(mol.n_atoms-1):
            if past_chainName != mol.chainName[i+1] or past_chainID != mol.chainID[i+1]:
                if mol.resName[i] in aa :
                    print("End of chain %s ; adding terminal residue to %s %i %s"% (past_chainID,mol.resName[i],mol.resNum[i],mol.atomName[i]))
                    resNum = mol.resNum[i]
                    j=0
                    while mol.resNum[i-j] ==resNum :
                        mol.resName[i - j] += "T"
                        j+=1
                else:
                    print("End of chain %s %s %i"% (past_chainID,mol.resName[i],mol.resNum[i]))
                past_chainName = mol.chainName[i+1]
                past_chainID = mol.chainID[i+1]


        i = mol.n_atoms-1
        if mol.resName[i] in aa:
            print("End of chain %s ; adding terminal residue to %s %i %s" % (
            past_chainID, mol.resName[i], mol.resNum[i], mol.atomName[i]))
            resNum = mol.resNum[i]
            j = 0
            while mol.resNum[i - j] == resNum:
                mol.resName[i - j] += "T"
                j += 1
        else:
            print("End of chain %s %s %i" % (past_chainID, mol.resName[i], mol.resNum[i]))

    def atom_res_reorder(mol):
        # Check res order :
        chains = list(set(mol.chainID))
        chains.sort()
        new_idx = []
        for c in chains:
            chain_idx = mol.get_chain(c)
            resNumlist = list(set(mol.resNum[chain_idx]))
            resNumlist.sort()
            for i in range(len(resNumlist)):
                idx = np.where(mol.resNum[chain_idx] == resNumlist[i])[0]
                new_idx += list(chain_idx[idx])
        mol.select_atoms(np.array(new_idx))

        # reorder atoms and res
        for c in chains:
            chain_idx = mol.get_chain(c)
            past_resNum = mol.resNum[chain_idx[0]]
            resNum = 1
            for i in range(len(chain_idx)):
                if mol.resNum[chain_idx[i]] != past_resNum:
                    past_resNum = mol.resNum[chain_idx[i]]
                    resNum += 1
                mol.resNum[chain_idx[i]] = resNum
                mol.atomNum[chain_idx[i]] = i + 1


    mol = Molecule(inputPDB)

    remove_alter_atom(mol)
    remove_hydrogens(mol)
    alias_atom(mol, "CD", "CD1", "ILE")
    alias_atom(mol, "OT1", "O")
    alias_atom(mol, "OT2", "OXT")
    alias_res(mol, "HSE", "HIS")
    alias_res(mol, "CYT", "C")
    alias_res(mol, "GUA", "G")
    alias_res(mol, "ADE", "A")
    alias_res(mol, "URA", "U")
    alias_atom(mol, "O1'", "O1*")
    alias_atom(mol, "O2'", "O2*")
    alias_atom(mol, "O3'", "O3*")
    alias_atom(mol, "O4'", "O4*")
    alias_atom(mol, "O5'", "O5*")
    alias_atom(mol, "C1'", "C1*")
    alias_atom(mol, "C2'", "C2*")
    alias_atom(mol, "C3'", "C3*")
    alias_atom(mol, "C4'", "C4*")
    alias_atom(mol, "C5'", "C5*")
    add_terminal_res(mol)
    atom_res_reorder(mol)

    mol.save_pdb(outputPDB)
