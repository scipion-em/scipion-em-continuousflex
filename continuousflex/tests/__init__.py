from .test_workflow_nma import *
from .test_workflow_nma3D import *

from pyworkflow.tests import DataSet

dic = dict(pdb='pdb/AK.pdb', particles='particles/*.spi', vol='volume/AK_LP10.vol',
           gold_atomic='gold/precomputed_with_atomic.xmd',
           gold_pseudoatomic='gold/precomputed_with_pseudoatomic.xmd')
DataSet(name='nma3D', folder='nma3D', files=dic)