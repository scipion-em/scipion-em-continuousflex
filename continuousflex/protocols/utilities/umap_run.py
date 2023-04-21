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

@numba.njit(nopython=True)
def rmsd2(a,b):
    mats = [[[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            [[0.5, 0.8660254, -0.],
             [-0.8660254, 0.5, 0.],
             [0., 0., 1.]],
            [[-0.5, 0.8660254, -0.],
             [-0.8660254, -0.5, 0.],
             [-0., 0., 1.]],
            [[-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0]],
            [[-0.5, -0.8660254, -0.],
             [0.8660254, -0.5, 0.],
             [-0., -0., 1.]],
            [[0.5, -0.8660254, -0.],
             [0.8660254, 0.5, 0.],
             [0., -0., 1.]]]

    coord1 = a
    natm = len(coord1) // 3
    nres = natm // 6
    chains = [1, 5, 4, 3, 2, 0]
    chains2 = chains
    out=[]
    for rot in range(6):
        dev = 0.0
        for i in range(6):
            for r in range(nres):
                x_coord = b[chains2[i] * nres + r]
                y_coord = b[chains2[i] * nres + r + natm]
                z_coord = b[chains2[i] * nres + r + (natm * 2)]
                if rot != 0:
                    x_coord_rot = mats[rot][0][0]*x_coord + mats[rot][1][0]*y_coord + mats[rot][2][0]*z_coord
                    y_coord_rot = mats[rot][0][1]*x_coord + mats[rot][1][1]*y_coord + mats[rot][2][1]*z_coord
                    z_coord_rot = mats[rot][0][2]*x_coord + mats[rot][1][2]*y_coord + mats[rot][2][2]*z_coord
                    x_coord = x_coord_rot
                    y_coord = y_coord_rot
                    z_coord = z_coord_rot
                dev += ((coord1[chains[i] * nres + r] - x_coord) ** 2 +
                           (coord1[chains[i] * nres + r + natm] - y_coord) ** 2 +
                           (coord1[chains[i] * nres + r + (natm * 2)] - z_coord) ** 2)
        out.append(np.sqrt(dev/natm))
        chains2 = chains2[1:] + [chains2[0]]

    return min(out)



def umap_run(n_component, n_neigbors, n_epocks, pkl_pdbs, pkl_pca, pkl_out,low_memory=True, metric_rmsd=False):
    pdbs_matrix = load(pkl_pdbs)
    if metric_rmsd:
        metric = rmsd2
        mat_reshape = pdbs_matrix.reshape(pdbs_matrix.shape[0],pdbs_matrix.shape[1],pdbs_matrix.shape[2])
        mat_reshape = np.transpose(mat_reshape, axis=(0,2,1))
        pdbs_matrix = mat_reshape.reshape(pdbs_matrix.shape[0],pdbs_matrix.shape[1]*pdbs_matrix.shape[2])
    else:
        metric= "euclidean"
    umap = UMAP(n_components=n_component, n_neighbors=n_neigbors, n_epochs=n_epocks,low_memory=low_memory,
                metric=metric).fit(pdbs_matrix)
    Y = umap.transform(pdbs_matrix)
    dump(umap, pkl_pca)
    dump(Y, pkl_out)

if __name__ == '__main__':
    umap_run(int(sys.argv[1]),
             int(sys.argv[2]),
             int(sys.argv[3]),
             sys.argv[4],
             sys.argv[5],
             sys.argv[6],
             bool(sys.argv[7]))
    sys.exit()
