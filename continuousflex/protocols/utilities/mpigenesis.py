import subprocess
from subprocess import Popen
import sys
import os
import shutil
import time

print("---------------- MPI GENESIS ------------------")
usage = "USAGE : \n"\
      "\t python mpigenesis.py \n" \
      "\t\t -c, --mpi_command MPI_COMMAND (mpi command to run e.g. \"mpirun -np 1\")\n"\
      "\t\t -p, --num_mpi NUM_MPI (total number of MPI cores available)\n"\
      "\t\t -t, --num_threads NUM_THREADS (Number of OMP threads)\n" \
      "\t\t -i, --inputs GENESIS_INPUT_FILES_PATH (file containing the path to the genesis inputs files to run)\n" \
      "\t\t -o, --outputs GENESIS_LOG_FILES_PATH (file containing the path to the output log files to use)\n" \
      "\t\t -e, --executable GENESIS_EXECUTABLE_PATH (path to genesis executable e.g. /path/to/atdyn)\n" \
      "\t\t [-a, --mpi_argument MPI_ARGUMENT ] (Additional arguments to pass to the MPI command)]\n" \
      "\t\t [-r, --rankdir RANK_DIRECTORY ] (If set, use rankfiles to attribute each run to a core)\n" \
      "\t\t [-cn, --num_core_per_node NUM_CORE_PER_NODE ] (if rankdir is set, defines the number of cores per node)\n" \
      "\t\t [-sn, --num_socket_per_node NUM_SOCKET_PER_NODE ] (if rankdir is set, defines the number of sockets per node)\n" \
      "\t\t [-n, --num_node NUM_NODE ] (if rankdir is set, defines the number of nodes) \n" \
      "\t\t [-l, --localhost ] (If set, use localhost instead of relative host)\n"\
      "\n\t\t -h, --help (Print this usage message)\n"

mpi_command = ""
num_mpi = 1
num_threads = 1
inputs_path = ""
outputs_path=""
executable=""

mpi_argument=None
localhost=False
use_rankfiles= False
rankdir = ""
num_core_per_node = 1
num_socket_per_node = 1
num_node = 1

for i in range(1, len(sys.argv)):
    if sys.argv[i] == "-h" or sys.argv[i] == "--help":
        print(usage)
        exit(0)
    if sys.argv[i] == "-c" or sys.argv[i] == "--mpi_command":
        mpi_command = sys.argv[i+1]
    if sys.argv[i] == "-p" or sys.argv[i] == "--num_mpi":
        num_mpi = int(sys.argv[i+1])
    elif sys.argv[i] == "-t" or sys.argv[i] == "--num_threads":
        num_threads = int(sys.argv[i+1])
    elif sys.argv[i] == "-i" or sys.argv[i] == "--inputs":
        inputs_path = sys.argv[i+1]
    elif sys.argv[i] == "-o" or sys.argv[i] == "--outputs":
        outputs_path = sys.argv[i+1]
    elif sys.argv[i] == "-e" or sys.argv[i] == "--executable":
        executable = sys.argv[i+1]
    elif sys.argv[i] == "-l" or sys.argv[i] == "--localhost":
        localhost = True
    elif sys.argv[i] == "-r" or sys.argv[i] == "--rankdir":
        rankdir = sys.argv[i + 1]
        use_rankfiles= True
    elif sys.argv[i] == "-a" or sys.argv[i] == "--mpi_argument":
        mpi_argument = sys.argv[i+1]
    elif sys.argv[i] == "-cn" or sys.argv[i] == "--num_core_per_node":
        num_core_per_node = int(sys.argv[i+1])
    elif sys.argv[i] == "-sn" or sys.argv[i] == "--num_socket_per_node":
        num_socket_per_node = int(sys.argv[i+1])
    elif sys.argv[i] == "-n" or sys.argv[i] == "--num_node":
        num_node = int(sys.argv[i + 1])


print("Parameters : ")
print("\t num_mpi -> %s"%num_mpi)
print("\t num_threads -> %s"%num_threads)
print("\t inputs -> %s"%inputs_path)
print("\t outputs -> %s"%outputs_path)
print("\t executable -> %s"%executable)
if mpi_argument is not None:
    print("\t mpi_argument -> %s" % mpi_argument)
if use_rankfiles :
    print("\t rankdir -> %s" % rankdir)
    print("\t num_node -> %s" % num_node)
    print("\t num_core_per_node -> %s" % num_core_per_node)
    print("\t num_socket_per_node -> %s" % num_socket_per_node)
    print("\t localhost -> %s"%str(localhost))

#Read inputs/ outputs
inputs = []
with open(inputs_path,"r") as f:
    for l in f:
        inputs.append(l.strip())
outputs = []
with open(outputs_path,"r") as f:
    for l in f:
        outputs.append(l.strip())
num_run = len(inputs)
if len(outputs) != num_run:
    raise RuntimeError("Error: number of inputs and outputs differs : %i != %i"%(num_run, len(outputs)))

num_core_per_socket = num_core_per_node//num_socket_per_node

if use_rankfiles:
    # Clean and rankfile dir
    if os.path.exists(rankdir) and len(rankdir):
            if os.path.isdir(rankdir):
                if os.path.islink(rankdir):
                    os.remove(rankdir)
                else:
                    shutil.rmtree(rankdir)
            else:
                os.remove(rankdir)
    os.makedirs(rankdir)
    # create rank files
    rankfiles = []
    for i in range(num_mpi):
        A = int( i/num_core_per_node )
        j = int( i%num_core_per_node)
        B= int(j/num_core_per_socket)
        C= int(j%24)
        if localhost:
            rank = "rank 0=localhost slot=%i:%i"%(B,C)
        else:
            rank = "rank 0=+n%i slot=%i:%i"%(A,B,C)
        rf = os.path.join(rankdir,"rank_file_%s"%str(i+1).zfill(6))
        with open(rf, "w") as f:
            f.write(rank)
        rankfiles.append(rf)

# prepare env
num_complete = 0
launch_index = 0
env = os.environ
env["OMP_NUM_THREADS"] = str(num_threads)
process = [None for i in range(num_mpi)]
status = [0 for i in range(num_mpi)]
if mpi_argument is None:
    mpi_argument = ""

# utils functions
def check_complete():
    complete = 0
    for i in range(num_mpi):
        p = process[i]
        if isinstance(p, subprocess.Popen):
            stat = p.poll()
            if stat is not None:
                if stat != status[i]:
                    status[i] = stat
                    complete +=1
                    if stat != 0 :
                        print("Warning : one task returned a non-zero exit")
            else:
                status[i] = stat
    return complete

def get_free_slot():
    for i in range(num_mpi):
        if status[i] is not None:
            return i
    return -1

def print_load():
    print("Task completed : %i / %i"%(num_complete, num_run))

print_load()
while (1):
    new = check_complete()
    num_complete += new
    if (num_complete == num_run):
        break
    if new !=0 :
        print_load()

    slot = get_free_slot()
    if slot == -1 :
        time.sleep(1)
    else:
        if launch_index < num_run:
            if use_rankfiles :
                rank_command = "--rankfile %s"%rankfiles[slot]
            else:
                rank_command = ""
            cmd = "%s %s %s %s %s > %s"\
                      % (mpi_command, rank_command, mpi_argument, executable, inputs[launch_index], outputs[launch_index])
            print(cmd)
            p = Popen(cmd, env=env, shell=True, cwd=os.getcwd())
            launch_index+=1
            process[slot] = p

print("All task completed")





