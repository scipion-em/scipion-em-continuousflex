# By Remi Vuillemot

import numpy as np
import copy
from Bio.SVDSuperimposer import SVDSuperimposer

class ContinuousFlexPDBHandler:
    
    @classmethod
    def read_coords(cls, pdb_file):
        print("> Reading pdb file %s ..." % pdb_file)
        coords = []
        with open(pdb_file, "r") as f:
            for line in f:
                if 'ATOM' in line:
                    coords.append([
                            line[30:38],line[38:46],line[46:54]
                        ])
        return np.array(coords).astype(float)

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
        insertion = []
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
                        l = [line[:6], line[6:11], line[12:16], line[16], line[17:21], line[21], line[22:26],
                             line[26], line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[72:76], line[76:78]]
                        l = [i.strip() for i in l]
                        atom.append(l[0])
                        atomNum.append(l[1])
                        atomName.append(l[2])
                        resAlter.append(l[3])
                        resName.append(l[4])
                        chainName.append(l[5])
                        resNum.append(l[6])
                        insertion.append(l[7])
                        coords.append([float(l[8]), float(l[9]), float(l[10])])
                        occ.append(l[11])
                        temp.append(l[12])
                        chainID.append(l[13])
                        elemName.append(l[14])

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
        self.insertion = np.array(insertion, dtype='<U1')
        self.coords = np.array(coords).astype(float)
        self.occ = np.array(occ).astype(float)
        self.temp = np.array(temp).astype(float)
        self.chainID = np.array(chainID, dtype='<U4')
        self.elemName = np.array(elemName, dtype='<U2')

        if self.n_atoms == 0 :
            raise RuntimeError("Could not read PDB file : PDB file is empty")

        print("\t Done \n")

    def write_pdb(self, file):
        """
        Write to PDB Format
        :param file: pdb file path
        """
        print("> Writing pdb file %s ..." % file)
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
                atomName = self.atomName[i].ljust(4)  # atomname$#4s
                resAlter = self.resAlter[i].ljust(1)  # resAlter#1
                resName = self.resName[i].ljust(4)  # resname#1s
                chainName = self.chainName[i].rjust(1)  # Astring
                resNum = str(self.resNum[i]).rjust(4)  # resnum
                insertion = self.insertion[i].ljust(1)  # resnum
                coordx = str('%8.3f' % (float(self.coords[i][0]))).rjust(8)  # x
                coordy = str('%8.3f' % (float(self.coords[i][1]))).rjust(8)  # y
                coordz = str('%8.3f' % (float(self.coords[i][2]))).rjust(8)  # z\
                occ = str('%6.2f' % self.occ[i]).rjust(6)  # occ
                temp = str('%6.2f' % self.temp[i]).rjust(6)  # temp
                chainID = str(self.chainID[i]).ljust(4)  # elname
                elemName = str(self.elemName[i]).rjust(2)  # elname
                file.write("%s%s %s%s%s%s%s%s   %s%s%s%s%s      %s%s\n" % (
                atom, atomNum, atomName, resAlter, resName, chainName, resNum,
                insertion, coordx, coordy, coordz, occ, temp, chainID, elemName))
            file.write("END\n")
        print("\t Done \n")

    def matchPDBatoms(self, reference_pdb, ca_only=False, matchingType=None):
        """
        match atoms between the pdb and a reference pdb
        :param reference_pdb: ContinuousflexPDBHandler
        :param ca_only: True if carbon alph only
        :param matchingType: 0= chain first, 1= segment ID first
        :return: index of matching atoms
        """
        print("> Matching PDBs atoms ...")
        n_mols = 2

        if matchingType == None:
            chain_name_list1 = self.get_chain_list(chainType=0)
            chain_name_list2 = reference_pdb.get_chain_list(chainType=0)
            n_matching_chain_names = sum([i in chain_name_list2 for i in chain_name_list1])
            print("Chains list 1  : "+str(chain_name_list1))
            print("Chains list 1  : "+str(chain_name_list2))
            print("Number of match : "+str(n_matching_chain_names))

            chain_id_list1 = self.get_chain_list(chainType=1)
            chain_id_list2 = reference_pdb.get_chain_list(chainType=1)
            n_matching_chain_ids = sum([i in chain_id_list2 for i in chain_id_list1])
            print("Seg list 1  : "+str(chain_id_list1))
            print("Seg list 1  : "+str(chain_id_list1))
            print("Number of match : "+str(n_matching_chain_ids))

            if n_matching_chain_ids == 0 and n_matching_chain_names == 0 :
                raise RuntimeError("No matching chains")
            elif n_matching_chain_ids >=n_matching_chain_names:
                matchingType = 1
                print("\t Matching segments %s ... "%n_matching_chain_ids)
            elif  n_matching_chain_ids < n_matching_chain_names:
                matchingType = 0
                print("\t Matching chains %s ... "%n_matching_chain_names)


        ids = []
        ids_idx = []
        for m in [self, reference_pdb]:
            id_tmp = []
            id_idx_tmp = []
            for i in range(m.n_atoms):
                if (not ca_only) or m.atomName[i] == "CA" or m.atomName[i] == "P":
                    id_tmp.append("%s_%i_%s_%s" % (m.chainName[i] if matchingType == 0 else m.chainID[i],
                                                   m.resNum[i], m.resName[i], m.atomName[i]))
                    id_idx_tmp.append(i)
            ids.append(np.array(id_tmp))
            ids_idx.append(np.array(id_idx_tmp))

        idx = []
        for i in range(len(ids[0])):
            idx_line = [ids_idx[0][i]]
            for m in range(1, n_mols):
                idx_tmp = np.where(ids[0][i] == ids[m])[0]
                if len(idx_tmp) == 1:
                    idx_line.append(ids_idx[m][idx_tmp[0]])
                elif len(idx_tmp) > 1:
                    print("\t Warning : One atom in mol#0 is matching several atoms in mol#%i : " % m)

            if len(idx_line) == n_mols:
                idx.append(idx_line)

        if len(idx) == 0:
            print("\t Warning : No matching coordinates")

        print("\t %i matching atoms " % len(np.array(idx)))
        print("\t Done")

        return np.array(idx)

    def alignMol(self, reference_pdb, idx_matching_atoms=None):
        print("> Aligning PDB ...")

        if idx_matching_atoms is not None:
            c1 = reference_pdb.coords[idx_matching_atoms[:, 1]]
            c2 = self.coords[idx_matching_atoms[:, 0]]
        else:
            c1 = reference_pdb.coords
            c2 = self.coords

        rot, tran = self.alignCoords(c1,c2)
        self_copy = self.copy()
        self_copy.coords = np.dot(self_copy.coords, rot) + tran
        print("\t Done \n")

        return self_copy

    @classmethod
    def alignCoords(cls, coord_ref, coord):
        sup = SVDSuperimposer()
        sup.set(coord_ref, coord)
        try:
            sup.run()
            rot, tran = sup.get_rotran()
        except np.linalg.LinAlgError:
            print("Error while aligning : SVD did not converge")
            rot = np.eye(3)
            tran = np.zeros(3)
        return rot, tran

    def getRMSD(self, reference_pdb, align=False, idx_matching_atoms=None):
        if align:
            aligned = self.alignMol(reference_pdb=reference_pdb, idx_matching_atoms=idx_matching_atoms)
        else:
            aligned=self
        if idx_matching_atoms is not None:
            coord1 = reference_pdb.coords[idx_matching_atoms[:, 1]]
            coord2 = aligned.coords[idx_matching_atoms[:, 0]]
        else:
            coord1 = reference_pdb.coords
            coord2 = aligned.coords
        return np.sqrt(np.mean(np.square(np.linalg.norm(coord1 - coord2, axis=1))))

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
        self.insertion = self.insertion[idx]
        self.elemName = self.elemName[idx]
        self.occ = self.occ[idx]
        self.temp = self.temp[idx]
        self.chainID = self.chainID[idx]

    def get_chain_list(self, chainType=0):
        if chainType == 0:
            lst = list(set(self.chainName))
        else:
            lst = list(set(self.chainID))
        lst.sort()
        return lst

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

    def copy(self):
        return copy.deepcopy(self)

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


    def check_res_order(self):
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


    def atom_res_reorder(self):
        chains = list(set(self.chainID))
        chains.sort()

        # reorder atoms and res
        for c in chains:
            chain_idx = self.get_chain(c)
            resNum = 1
            previousRes = str(self.resNum[chain_idx[0]]) + self.insertion[chain_idx[0]]
            for i in range(len(chain_idx)):
                currentRes = str(self.resNum[chain_idx[i]]) + self.insertion[chain_idx[i]]

                if currentRes != previousRes:
                    previousRes = currentRes
                    resNum +=1
                self.resNum[chain_idx[i]] = resNum
                self.atomNum[chain_idx[i]] = i + 1

        #remove insertions :
        for i in range(self.n_atoms) :
            self.insertion[i] = " "


    def allatoms2ca(self):
        new_idx = []
        for i in range(self.n_atoms):
            if self.atomName[i] == "CA" or self.atomName[i] == "P":
                new_idx.append(i)
        return np.array(new_idx)

    def center(self):
        self.coords -= np.mean(self.coords, axis=0)

