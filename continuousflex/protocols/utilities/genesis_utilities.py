# By Remi Vuillemot

import numpy as np
from pyworkflow.utils import runCommand
import multiprocessing

NUMBER_OF_CPU = int(np.min([multiprocessing.cpu_count(),4]))

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

INPUT_TOPOLOGY = 0
INPUT_RESTART = 1
INPUT_NEW_SIM = 2

PROJECTION_ANGLE_SAME=0
PROJECTION_ANGLE_XMIPP=1
PROJECTION_ANGLE_IMAGE=2


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
                    if len(splitline) >= len(header):
                        for i in range(1,len(header)):
                            try :
                                dic[header[i]].append(float(splitline[i]))
                            except ValueError:
                                pass
    return dic
def dcd2numpyArr(filename):
    print("> Reading dcd file %s"%filename)
    BYTESIZE = 4
    with open(filename, 'rb') as f:

        # Header
        # ---------------- INIT

        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        crd_type = f.read(BYTESIZE).decode('ascii')
        nframe = int.from_bytes((f.read(BYTESIZE)), "little")
        start_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_total = int.from_bytes((f.read(BYTESIZE)), "little")
        for i in range(5):
            f.read(BYTESIZE)
        time_step = np.frombuffer(f.read(BYTESIZE), dtype=np.float32)
        for i in range(9):
            f.read(BYTESIZE)
        charmm_version = int.from_bytes((f.read(BYTESIZE)), "little")

        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- TITLE
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        ntitle = int.from_bytes((f.read(BYTESIZE)), "little")
        tilte_rd = f.read(BYTESIZE*20 * ntitle)
        try :
            title = tilte_rd.encode("ascii")
        except AttributeError:
            title = str(tilte_rd)
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- NATOM
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        natom = int.from_bytes((f.read(BYTESIZE)), "little")
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ----------------- DCD COORD
        dcd_arr =  np.zeros((nframe, natom, 3), dtype=np.float32)
        for i in range(nframe):
            for j in range(3):

                start_size = int.from_bytes((f.read(BYTESIZE)), "little")
                while (start_size != BYTESIZE * natom and start_size != 0):
                    # print("\n-- UNKNOWN %s -- " % start_size)

                    f.read(start_size)
                    end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                    if end_size != start_size:
                        raise RuntimeError("Can not read dcd file")
                    start_size = int.from_bytes((f.read(BYTESIZE)), "little")

                bin_arr = f.read(BYTESIZE * natom)
                if len(bin_arr) == BYTESIZE * natom:
                    dcd_arr[i, :, j] = np.frombuffer(bin_arr, dtype=np.float32)
                else:
                    break
                end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                if end_size != start_size:
                    if i>1:
                        break
                    else:
                        # pass
                        raise RuntimeError("Can not read dcd file %i %i " % (start_size, end_size))

        print("\t -- Summary of DCD file -- ")
        print("\t\t crd_type  : %s"%crd_type)
        print("\t\t nframe  : %s"%nframe)
        print("\t\t len_frame  : %s"%len_frame)
        print("\t\t len_total  : %s"%len_total)
        print("\t\t time_step  : %s"%time_step)
        print("\t\t charmm_version  : %s"%charmm_version)
        print("\t\t title  : %s"%title)
        print("\t\t natom  : %s"%natom)
    print("\t Done \n")

    return dcd_arr

def numpyArr2dcd(arr, filename, start_frame=1, len_frame=1, time_step=1.0, title=None):
    print("> Wrinting dcd file %s"%filename)
    BYTESIZE = 4
    nframe, natom, _ = arr.shape
    len_total=nframe*len_frame
    charmm_version=24
    if title is None:
        title = "DCD file generated by Continuous Flex plugin"
    ntitle = (len(title)//(20*BYTESIZE)) + 1
    with open(filename, 'wb') as f:
        zeroByte = int.to_bytes(0, BYTESIZE, "little")

        # Header
        # ---------------- INIT
        f.write(int.to_bytes(21*BYTESIZE ,BYTESIZE, "little"))
        f.write(b'CORD')
        f.write(int.to_bytes(nframe, BYTESIZE, "little"))
        f.write(int.to_bytes(start_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_total, BYTESIZE, "little"))
        for i in range(5):
            f.write(zeroByte)
        f.write(np.float32(time_step).tobytes())
        for i in range(9):
            f.write(zeroByte)
        f.write(int.to_bytes(charmm_version, BYTESIZE, "little"))

        f.write(int.to_bytes(21*BYTESIZE,BYTESIZE, "little"))

        # ---------------- TITLE
        f.write(int.to_bytes((ntitle*20+1)*BYTESIZE ,BYTESIZE, "little"))
        f.write(int.to_bytes(ntitle ,BYTESIZE, "little"))
        f.write(title.ljust(20*BYTESIZE).encode("ascii"))
        f.write(int.to_bytes((ntitle*20+1)*BYTESIZE ,BYTESIZE, "little"))

        # ---------------- NATOM
        f.write(int.to_bytes(BYTESIZE ,BYTESIZE, "little"))
        f.write(int.to_bytes(natom ,BYTESIZE, "little"))
        f.write(int.to_bytes(BYTESIZE ,BYTESIZE, "little"))

        # ----------------- DCD COORD
        for i in range(nframe):
            for j in range(3):
                f.write(int.to_bytes(BYTESIZE*natom, BYTESIZE, "little"))
                f.write(np.float32(arr[i, :, j]).tobytes())
                f.write(int.to_bytes(BYTESIZE*natom, BYTESIZE, "little"))
    print("\t Done \n")

def existsCommand(name):
    from shutil import which
    return which(name) is not None
