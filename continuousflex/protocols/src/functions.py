import hashlib
import multiprocessing.pool
import os.path
import pickle

import autograd.numpy as npg
import matplotlib.pyplot as plt
import numpy as np

from continuousflex.protocols.src.flexible_fitting import FlexibleFitting

def get_RMSD(psim, pexp):
    return np.linalg.norm(psim-pexp)**2

def select_voxels(coord, size, sampling_rate, threshold):
    n_atoms = coord.shape[0]
    n_vox = threshold*2 +1
    l=np.zeros((n_atoms,3))

    for i in range(n_atoms):
        l[i] = (coord[i]/sampling_rate -threshold + size/2).astype(int)

    if (np.max(l) >= size or np.min(l)<0):
        raise RuntimeError("threshold too large")
    return l.astype(int), n_vox

def to_vector(arr):
    X,Y,Z = arr.shape
    vec = np.zeros(X*Y*Z)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                vec[z + y*Z + x*Y*Z] = arr[x,y,z]
    return vec

def to_matrix(vec, X,Y,Z):
    arr = np.zeros((X,Y,Z))
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                arr[x, y, z] = vec[z + y * Z + x * Y * Z]
    return arr

def cross_correlation(map1, map2):
    return np.sum(map1*map2)/np.sqrt(np.sum(np.square(map1))*np.sum(np.square(map2)))

def generate_euler_matrix(angles):
    a, b, c = angles
    cos = npg.cos
    sin = npg.sin
    R = npg.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
                  [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
                  [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
    return R

def get_euler_grad(angles, coord):
    a,b,c = angles
    x, y, z = coord
    cos = np.cos
    sin = np.sin

    dR = np.array([[x* (cos(c) *  cos(b) * -sin(a) -  sin(c) * cos(a)) + y* ( cos(c) * cos(b) * cos(a) +  sin(c) * -sin(a)),
                    x* (- sin(c) *  cos(b) * -sin(a) - cos(c) * cos(a)) + y* ( - sin(c) * cos(b) * cos(a) + cos(c) * -sin(a)),
                    x* (sin(b) * -sin(a)) + y* (sin(b) * cos(a))],

                   [x* (cos(c) * -sin(b) * cos(a)) + y* ( cos(c) * -sin(b) * sin(a)) + z* ( -cos(c) * cos(b)),
                    x* (- sin(c) *  -sin(b) * cos(a)) + y* ( - sin(c) * -sin(b) * sin(a)) + z* (sin(c) * cos(b)),
                    x* (cos(b) * cos(a)) + y* (cos(b) * sin(a)) + z* (-sin(b))],

                   [x* (-sin(c) *  cos(b) * cos(a) -  cos(c) * sin(a)) + y* ( -sin(c) * cos(b) * sin(a) +  cos(c) * cos(a)) + z* ( sin(c) * sin(b)),
                    x * (- cos(c) * cos(b) * cos(a) + sin(c) * sin(a)) + y* ( - cos(c) * cos(b) * sin(a) - sin(c) * cos(a))+ z* (cos(c) * sin(b)),
                    0]])

    return dR

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

def multiple_fitting(init, targets, vars, n_chain, n_iter, n_warmup, params, n_proc, verbose):
    ff = []
    N = len(targets)
    for t in targets :
        ff.append(FlexibleFitting(init=init, target = t, vars=vars, n_chain=n_chain,
                                  n_iter=n_iter, n_warmup=n_warmup, params=params, verbose=verbose))
    ff = np.array(ff)

    n_loop = (N * n_chain) //n_proc
    n_last_process = ((N * n_chain) % n_proc)//n_chain
    n_process = n_proc//n_chain
    process = [np.arange(i*n_process, (i+1)*n_process) for i in range(n_loop)]
    process.append(np.arange(n_loop*n_process, n_loop*n_process + n_last_process))
    fits=[]
    print("Number of loops : "+str(n_loop))
    for i in process:

        print("\t fitting models # "+str(i))
        try :
            with NestablePool(n_process) as p:
                fits += p.map(FlexibleFitting.HMC, ff[i])
                p.close()
                p.join()
        except RuntimeError:
            print("Failed")
    return fits

