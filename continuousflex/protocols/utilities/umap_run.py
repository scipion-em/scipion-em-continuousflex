# By Mohamad Harastani and Remi Vuillemot

from umap import UMAP
import sys
from joblib import load, dump
import numba
import numpy as np

@numba.njit()
def rmsd(a,b):
    natoms = len(a) //3
    return np.sqrt(np.mean(
        np.square(a[:natoms] - b[:natoms]) +
        np.square(a[natoms:(natoms*2)] - b[natoms:(natoms*2)]) +
        np.square(a[(natoms*2):(natoms*3)] - b[(natoms*2):(natoms*3)])
    ))

def umap_run(n_component, n_neigbors, n_epocks, pkl_pdbs, pkl_pca, pkl_out,low_memory=True, metric_rmsd=False):
    pdbs_matrix = load(pkl_pdbs)
    if metric_rmsd:
        metric = rmsd
        mat_reshape = pdbs_matrix.reshape(pdbs_matrix.shape[0],pdbs_matrix.shape[1],pdbs_matrix.shape[2])
        mat_reshape = np.transpose(mat_reshape, axis=(0,2,1))
        pdbs_matrix = mat_reshape.reshape(pdbs_matrix.shape[0],pdbs_matrix.shape[1]*pdbs_matrix.shape[2])
    else:
        metric= "euclidean"
    umap = UMAP(n_components=n_component, n_neighbors=n_neigbors, n_epochs=n_epocks,low_memory=low_memory,
                metric=metric).fit(pdbs_matrix)
    Y = umap.transform(pdbs_matrix)
    dump(umap, pkl_pca)
    np.savetxt(pkl_out, Y)

if __name__ == '__main__':

    print("inputs : ")
    for i in sys.argv :
        print(i)
    umap_run(int(sys.argv[1]),
             int(sys.argv[2]),
             int(sys.argv[3]),
             sys.argv[4],
             sys.argv[5],
             sys.argv[6],
             bool(int(sys.argv[7])),
             bool(int(sys.argv[8])))
    sys.exit()
