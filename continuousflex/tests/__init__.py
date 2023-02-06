from .test_workflow_HEMNMA import *
from .test_workflow_HEMNMA3D import *
from .test_workflow_StA import *
from .test_structure_mapping import *
from .test_workflow_subtomogram_synthesize import *
from .test_workflow_TomoFlow import *
from .test_workflow_GENESIS import *
from .test_workflow_Deep_HEMNMA import *
from pyworkflow.tests import DataSet

files_dictionary = {'pdb': 'pdb/AK.pdb', 'particles': 'particles/img.stk', 'vol': 'volumes/AK_LP10.vol',
                    'precomputed_atomic': 'gold/images_WS_atoms.xmd',
                    'precomputed_pseudoatomic': 'gold/images_WS_pseudoatoms.xmd',
                    'small_stk': 'test_alignment_10images/particles/smallstack_img.stk',
                    'subtomograms':'HEMNMA_3D/subtomograms/*.vol',
                    'precomputed_HEMNMA3D_atoms':'HEMNMA_3D/gold/precomputed_atomic.xmd',
                    'precomputed_HEMNMA3D_pseudo':'HEMNMA_3D/gold/precomputed_pseudo.xmd',

                    'charmm_prm':'genesis/par_all36_prot.prm',
                    'charmm_top':'genesis/top_all36_prot.rtf',
                    '1ake_pdb':'genesis/1ake.pdb',
                    '1ake_vol':'genesis/1ake.mrc',
                    '4ake_pdb':'genesis/4ake.pdb',
                    '4ake_aa_pdb':'genesis/4ake_aa.pdb',
                    '4ake_aa_psf':'genesis/4ake_aa.psf',
                    '4ake_ca_pdb':'genesis/4ake_ca.pdb',
                    '4ake_ca_top':'genesis/4ake_ca.top',
                    }
DataSet(name='nma_V2.0', folder='nma_V2.0', files=files_dictionary,
        url='https://raw.githubusercontent.com/continuousflex-org/testdata-continuousflex/main')
