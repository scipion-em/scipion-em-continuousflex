# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@igbmc.fr)
# * Remi Vuillemot (remi.vuillemot@upmc.fr)
# * Ilyes Hamitouche (ilyes.hamitouche@upmc.fr)
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
# *  e-mail address 'scipion@cnb.csic.es' (if scipion related)
# *  e-mail address 'slavica.jonic@upmc.fr' (for methods issues)
# **************************************************************************
import os
import pwem
from continuousflex.constants import *
import pyworkflow.utils as pwutils
getXmippPath = pwem.Domain.importFromPlugin("xmipp3.base", 'getXmippPath')
import datetime
from scipion.install.funcs import VOID_TGZ
import continuousflex

_logo = "logo.png"

MD_NMMD_GENESIS_VERSION = "1.1"
# Use this variable to activate an environment from the Scipion conda
MODEL_CONTINUOUSFLEX_ENV_ACTIVATION_VAR = "MODEL_CONTINUOUSFLEX_ENV_ACTIVATION"
# Use this general activation variable when installed outside Scipion
MODEL_CONTINUOUSFLEX_ACTIVATION_VAR = "MODEL_CONTINUOUSFLEX_ACTIVATION"
CF_VERSION = 'git'

__version__ = "3.3.3"

class Plugin(pwem.Plugin):
    _homeVar = CONTINUOUSFLEX_HOME
    _pathVars = [CONTINUOUSFLEX_HOME]
    _supportedVersions = [VV]
    _url = CONTINUOUSFLEX_URL

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(MODEL_CONTINUOUSFLEX_ACTIVATION_VAR, '')
        cls._defineVar(MODEL_CONTINUOUSFLEX_ENV_ACTIVATION_VAR, cls.getActivationCmd(CF_VERSION))
        cls._defineEmVar(CONTINUOUSFLEX_HOME, continuousflex.__path__[0])
        cls._defineEmVar(NMA_HOME,'nma')
        cls._defineEmVar(GENESIS_HOME, 'MD-NMMD-Genesis-'+MD_NMMD_GENESIS_VERSION)
        cls._defineVar(VMD_HOME,'/usr/local/lib/vmd')
        cls._defineVar(MATLAB_HOME, '~/programs/Matlab')

    # TODO: These were copied from Xmipp, and we need to review if they are still needed here
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
    def getContinuousFlexCmd(cls, args):
        cmd = cls.getVar(MODEL_CONTINUOUSFLEX_ACTIVATION_VAR)
        if not cmd:
            cmd = cls.getCondaActivationCmd()
            cmd += cls.getVar(MODEL_CONTINUOUSFLEX_ENV_ACTIVATION_VAR)
        cmd += " && "
        cmd += args
        return cmd

    @classmethod
    def getActivationCmd(cls, version):
        return 'conda activate continuousflex-' + version

    @classmethod
    def isVersionActive(cls):
        return cls.getActiveVersion().startswith(VV)

    @classmethod
    def defineBinaries(cls, env):
        os.environ['PATH'] += os.pathsep + env.getBinFolder()

        def defineCondaInstallation(version):
            installed = "last-pull-%s.txt" % datetime.datetime.now().strftime("%y%h%d-%H%M%S")

            cf_commands = []
            cf_commands.append((getCondaInstallation(version), 'env-created.txt'))

            env.addPackage('ContinuousFlex', version=version,
                           commands=cf_commands,
                           tar=VOID_TGZ,
                           default=True)

        def getCondaInstallation(version):
            installationCmd = cls.getCondaActivationCmd()
            config_path = continuousflex.__path__[0]+'/conda.yaml'
            installationCmd += 'conda env create -f {} --force -n continuousflex-'.format(config_path) + version + ' && '
            installationCmd += 'touch env-created.txt'
            return installationCmd

        # Install the conda environment with lapack and arpack
        defineCondaInstallation(CF_VERSION)

        # Cleaning the nma binaries files and folder before expanding
        if os.path.exists(env.getEmFolder() + '/nma*.tgz'):
            os.system('rm ' + env.getEmFolder() + '/nma*.tgz')


        cmd_1 = cls.getCondaActivationCmd() + ' ' + cls.getActivationCmd(CF_VERSION)
        cmd = cmd_1 + ' && cd ElNemo; make; mv nma_* ..'
        # TODO: if gcc, mpi and fortran are installed on the system, then these ljnes can be used to override their banaries
        # 'ln -s $GCC "$(dirname "${GCC}")"/gcc'
        # 'ln -s $GXX "$(dirname "${GXX}")"/gxx'
        # 'ln -s $(which x86_64-conda-linux-gnu-gfortran) "$(dirname "$(which x86_64-conda-linux-gnu-gfortran)")"/gfortran'

        lib_path = os.environ['CONDA_PREFIX_1'] + '/envs/continuousflex-' + CF_VERSION + '/lib'
        # linking blas, arpack and lapack libraries to scipion lib
        os.system('ln -f -s ' + lib_path + '/libopenblas* ' + env.getLibFolder())
        os.system('ln -f -s ' + lib_path + '/libarpack* ' + env.getLibFolder())
        os.system('ln -f -s ' + lib_path + '/liblapack* ' + env.getLibFolder())

        env.addPackage('nma', version='3.1',
                       url='https://github.com/continuousflex-org/NMA_basic_code/raw/master/nma_v5.tar',
                       createBuildDir=False,
                       buildDir='nma',
                       target="nma",
                       commands=[(cmd ,'nma_elnemo_pdbmat'),
                                 ('cd NMA_cart; LDFLAGS=-L%s make; mv nma_* ..'
                                  % lib_path, 'nma_diag_arpack')],
                       neededProgs=['gfortran'], default=True)

        cmd = cmd_1 + ' && pip install -U torch==1.10.1 torchvision==0.11.2 tensorboard==2.8.0 tqdm==4.64.0' \
                      ' protobuf==3.20.3' \
                      ' && touch DeepLearning_Installed'
        env.addPackage('DeepLearning', version='1.0',
                       tar='void.tgz',
                       buildDir='DeepLearning',
                       commands=[(cmd ,'DeepLearning_Installed')],
                       default=True)

        cmd = cmd_1 + ' && pip install -U setuptools==63.4.3 pycuda==2020.1 farneback3d==0.1.3' \
                      ' && touch OpticalFlow_Installed'
        env.addPackage('OpticalFlow', version='1.0',
                       tar='void.tgz',
                       commands=[(cmd,'OpticalFlow_Installed')],
                       neededProgs=[''],
                       default=True)

        target_branch = "merge_genesis_1.4"
        cmd = cmd_1 + ' && git clone -b %s https://github.com/continuousflex-org/MD-NMMD-Genesis.git . ; autoreconf -fi ;' \
                      ' ./configure LDFLAGS=-L\"%s\" FFLAGS=\"-fallow-argument-mismatch -ffree-line-length-none\";' \
                      ' make install;' % (target_branch, lib_path)

        env.addPackage('MD-NMMD-Genesis', version=MD_NMMD_GENESIS_VERSION,
                       buildDir='MD-NMMD-Genesis', tar="void.tgz",
                       commands=[(cmd , ["bin/atdyn"])],
                       neededProgs=['mpif90'], default=False)
