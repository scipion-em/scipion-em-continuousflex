# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@upmc.fr)
# * Slavica Jonic (slavica.jonic@upmc.fr)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import os

import pwem
from continuousflex.constants import *
import pyworkflow.utils as pwutils
getXmippPath = pwem.Domain.importFromPlugin("xmipp3.base", 'getXmippPath')
from pyworkflow.tests import DataSet

_logo = "logo.png"
__version__ = "3.0.33"

class Plugin(pwem.Plugin):
    _homeVar = CONTINUOUSFLEX_HOME
    _pathVars = [CONTINUOUSFLEX_HOME]
    _supportedVersions = [VV]

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(CONTINUOUSFLEX_HOME, 'xmipp')
        cls._defineEmVar(NMA_HOME,'nma')
        cls._defineEmVar(GENESIS_HOME, 'genesis-1.4.0')
        cls._defineEmVar(SITUS_HOME, 'situs-3.1')
        cls._defineVar(VMD_HOME,'/usr/local/lib/vmd')

    #   @classmethod
    #   def getEnviron(cls):
    #       """ Setup the environment variables needed to launch the program. """
    #      environ = Environ(os.environ)
    #       environ.update({
    #            'PATH': Plugin.getHome(),
    #        }, position=Environ.BEGIN)
    #
    #       return environ

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END
        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
            'PYTHONPATH': getXmippPath('pylib')
        }, position=pos)

        # environ variables are strings not booleans
        if os.environ.get('CUDA', 'False') != 'False':
            environ.update({
                'PATH': os.environ.get('CUDA_BIN', ''),
                'LD_LIBRARY_PATH': os.environ.get('NVCC_LIBDIR', '')
            }, position=pos)

        return environ


    @classmethod
    def isVersionActive(cls):
        return cls.getActiveVersion().startswith(VV)

    @classmethod
    def defineBinaries(cls, env):
        os.environ['PATH'] += os.pathsep + env.getBinFolder()
        lapack = env.addLibrary(
            'lapack',
            tar='lapack-3.5.0.tgz',
            flags=['-DBUILD_SHARED_LIBS:BOOL=ON',
                   '-DLAPACKE:BOOL=ON'],
            cmake=True,
            neededProgs=['gfortran'],
            default=False)

        arpack = env.addLibrary(
            'arpack',
            tar='arpack-96.tgz',
            neededProgs=['gfortran'],
            commands=[('cd ' + env.getBinFolder() + '; ln -s $(which gfortran) f77',
                       env.getBinFolder() + '/f77'),
                      ('cd ' + env.getTmpFolder() + '/arpack-96; make all',
                       env.getLibFolder() + '/libarpack.a')])
        # See http://modb.oce.ulg.ac.be/mediawiki/index.php/How_to_compile_ARPACK

        # Cleaning the nma binaries files and folder before expanding
        if os.path.exists(env.getEmFolder() + '/nma-2.0.tgz'):
            os.system('rm ' + env.getEmFolder() + '/nma-2.0.tgz')

        env.addPackage('nma', version='3.0', deps=[arpack, lapack],
                       # url='https://github.com/slajo/NMA_basic_code/raw/master/nma_v3.tar',
                       url='https://github.com/MohamadHarastani/nma_basic_codes/raw/main/nma_v4.tar',
                       createBuildDir=False,
                       buildDir='nma',
                       target="nma",
                       commands=[('cd ElNemo; make; mv nma_* ..',
                                  'nma_elnemo_pdbmat'),
                                 ('cd NMA_cart; LDFLAGS=-L%s make; mv nma_* ..'
                                  % env.getLibFolder(), 'nma_diag_arpack')],
                       neededProgs=['gfortran'], default=True)

        if os.path.exists(env.getEmFolder() + '/genesis.tgz'):
            os.system('rm ' + env.getEmFolder() + '/genesis.tgz')

        env.addPackage('genesis', version='1.4.0', deps=[lapack],
                       url='https://github.com/mms29/nmmd/archive/nmmd.tar.gz',
                       tar='genesis.tgz',
                       createBuildDir=True,
                       buildDir='genesis',
                       commands=[('tar -xf ../genesis.tgz -C .;'
                                  'mv nmmd-nmmd/* .;'
                                  'rm -r nmmd-nmmd;'
                                  './configure;'
                                  'make install;', "bin/atdyn")],
                       neededProgs=['mpif90'],
                       target="genesis", default=True)

files_dictionary = {'pdb': 'pdb/AK.pdb', 'particles': 'particles/img.stk', 'vol': 'volumes/AK_LP10.vol',
                    'precomputed_atomic': 'gold/images_WS_atoms.xmd',
                    'precomputed_pseudoatomic': 'gold/images_WS_pseudoatoms.xmd',
                    'small_stk': 'test_alignment_10images/particles/smallstack_img.stk',
                    'subtomograms':'HEMNMA_3D/subtomograms/*.vol',
                    'precomputed_HEMNMA3D_atoms':'HEMNMA_3D/gold/precomputed_atomic.xmd',
                    'precomputed_HEMNMA3D_pseudo':'HEMNMA_3D/gold/precomputed_pseudo.xmd',
                    'charmm_prm':'genesis/par_all36_prot.prm',
                    'charmm_top':'genesis/top_all36_prot.rtf',
                    'charmm_str':'genesis/toppar_water_ions.str',
                    '1ake_pdb':'genesis/1ake.pdb',
                    '1ake_vol':'genesis/1ake.vol',
                    '4ake_pdb':'genesis/4ake.pdb',
                    'ionize_pdb':'genesis/ionize.pdb',
                    'ionize_psf':'genesis/ionize.psf',
                    '4ake_ca_pdb':'genesis/4ake_cago.pdb',
                    '4ake_ca_top':'genesis/4ake_cago.top',
                    }
DataSet(name='nma_V2.0', folder='nma_V2.0', files=files_dictionary,
        url='https://raw.githubusercontent.com/MohamadHarastani/nma_V2.0/main/')

