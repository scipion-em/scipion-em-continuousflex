# By Mohamad Harastani and Remi Vuillemot

from umap import UMAP
import sys
from joblib import load, dump

def umap_run(n_component, n_neigbors, n_epocks, pkl_pdbs, pkl_pca, pkl_out):
    pdbs_matrix = load(pkl_pdbs)
    umap = UMAP(n_components=n_component, n_neighbors=n_neigbors, n_epochs=n_epocks).fit(pdbs_matrix)
    Y = umap.transform(pdbs_matrix)
    dump(umap, pkl_pca)
    dump(Y, pkl_out)

if __name__ == '__main__':
    umap_run(int(sys.argv[1]),
             int(sys.argv[2]),
             int(sys.argv[3]),
             sys.argv[4],
             sys.argv[5],
             sys.argv[6])
    sys.exit()
