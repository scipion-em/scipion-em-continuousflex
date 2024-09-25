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
import datetime
from scipion.install.funcs import VOID_TGZ
import continuousflex
import subprocess
import re
import pyworkflow.utils as pwutils


_logo = "logo.png"

MD_NMMD_GENESIS_VERSION = "2.1"
# Use this variable to activate an environment from the Scipion conda
MODEL_CONTINUOUSFLEX_ENV_ACTIVATION_VAR = "MODEL_CONTINUOUSFLEX_ENV_ACTIVATION"
# Use this general activation variable when installed outside Scipion
MODEL_CONTINUOUSFLEX_ACTIVATION_VAR = "MODEL_CONTINUOUSFLEX_ACTIVATION"

__version__ = "3.4.2"


class Plugin(pwem.Plugin):
    _homeVar = CONTINUOUSFLEX_HOME
    _pathVars = [CONTINUOUSFLEX_HOME]
    # We only support our latest release, we can't afford supporting previous releases
    _supportedVersions = [__version__]
    _url = CONTINUOUSFLEX_URL

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(MODEL_CONTINUOUSFLEX_ACTIVATION_VAR, '')
        cls._defineEmVar(CONTINUOUSFLEX_HOME, 'ContinuousFlex-' + __version__)
        cls._defineVar(MODEL_CONTINUOUSFLEX_ENV_ACTIVATION_VAR, cls.getActivationCmd(__version__))
        cls._defineEmVar(NMA_HOME, 'nma')
        cls._defineEmVar(GENESIS_HOME, 'MDTools-' + MD_NMMD_GENESIS_VERSION)
        cls._defineEmVar(SMOG_HOME, 'smog-2.4.5')
        cls._defineVar(VMD_HOME, '/usr/local/lib/vmd')
        cls._defineVar(MATLAB_HOME, '~/programs/Matlab')

    @classmethod
    def getEnviron(cls):
        environ = pwutils.Environ(os.environ)
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
        return 'conda activate {}'.format(cls.getVar(CONTINUOUSFLEX_HOME))

    @classmethod
    def isVersionActive(cls):
        return cls.getActiveVersion().startswith(__version__)

    @classmethod
    def getCondaLibPath(cls):
        return cls.getVar(CONTINUOUSFLEX_HOME) + '/lib'

    @classmethod
    def defineBinaries(cls, env):
        os.environ['PATH'] += os.pathsep + env.getBinFolder()

        def defineCondaInstallation(version):
            installed = "last-pull-%s.txt" % __version__

            cf_commands = []
            cf_commands.append((getCondaInstallation(version, installed), installed))

            env.addPackage('ContinuousFlex', version=version,
                           commands=cf_commands,
                           tar=VOID_TGZ,
                           default=True)

        def getCondaInstallation(version, txtfile):
            installationCmd = cls.getCondaActivationCmd()
            config_path = continuousflex.__path__[0] + '/conda_noCuda.yaml'
            installationCmd += 'conda env create -f {} --prefix . --yes'.format(config_path)
            # If nvcc is in the path, install Optical Flow and DeepLearning Libraries
            if os.popen('which nvcc').read() is not None:
                config_path = continuousflex.__path__[0] + '/conda.yaml'
                installationCmd += "&& conda env update --prefix . --file={}".format(config_path)
            installationCmd += ' && touch {}'.format(txtfile)
            return installationCmd

        # Install the conda environment followed by the binaries
        defineCondaInstallation(__version__)
        env.addPackage('nma', version='3.1',
                       url='https://github.com/continuousflex-org/NMA_basic_code/raw/master/nma_v5.tar',
                       createBuildDir=False,
                       buildDir='nma',
                       target="nma",
                       commands=[('cd ElNemo; make; mv nma_* ..', 'nma_elnemo_pdbmat'),
                                 ('cd NMA_cart; LDFLAGS=-L%s make; mv nma_* ..'
                                  % cls.getCondaLibPath() , 'nma_diag_arpack')],
                       neededProgs=['gfortran'], default=True)

        target_branch = "master"
        output = subprocess.getoutput("gfortran --version")
        gfotran_version = int(re.search(r'\d+', output).group())
        if gfotran_version >= 10:
            FFLAGS = "-fallow-argument-mismatch -ffree-line-length-none"
        else:
            FFLAGS = "-ffree-line-length-none"

        env.addPackage('MDTools', version=MD_NMMD_GENESIS_VERSION,
                       buildDir='MDTools', tar="void.tgz",
                       commands=[(
                           'git clone -b %s https://github.com/continuousflex-org/MDTools.git . &&'
                           'mkdir lib && cp %s/libopenblas* lib && cp %s/libblas* lib && cp %s/liblapack* lib &&'
                           ' autoreconf -fi && ./configure LDFLAGS=-L%s/lib FFLAGS=\"%s\" && make install;'
                           % (target_branch, cls.getCondaLibPath(),
                              cls.getCondaLibPath(),cls.getCondaLibPath(), cls.getVar("GENESIS_HOME"), FFLAGS), ["bin/atdyn"])],
                       neededProgs=['mpif90'], default=True)


        env.addPackage('smog', version="2.4.5",
                       buildDir='smog-2.4.5', url="https://smog-server.org/smog2/code/smog-2.4.5.tgz",
                       target="smog-2.4.5",
                       commands=[( "mkdir -p smogenv && cd smogenv && %s conda env create -f %s/smog2.yaml --yes --prefix . " 
                                  "&& cd .. && %s/smog-2.4.5//smogenv/bin/perl -MCPAN -e 'install XML::Validator::Schema' &&"
                                  "export perl4smog=\"%s/smog-2.4.5/smogenv/bin/perl\" && "
                                  "echo -n '#!/bin/bash' > configure && "
                                  "echo "" >> configure &&"
                                  "cat configure.smog2 >> configure &&"
                                  "chmod 777 configure &&"
                                  "./configure"%
                                  (cls.getCondaActivationCmd(),continuousflex.__path__[0], env.getEmFolder(), env.getEmFolder()),
                                   ["bin/smog2"])], default=True)
