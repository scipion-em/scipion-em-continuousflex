# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Slavica Jonic                (slavica.jonic@upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Jan 2014
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

from pwem import *
from pwem.emlib import (MetaData, MDL_X, MDL_COUNT, MDL_NMA_MODEFILE, MDL_ORDER,
                        MDL_ENABLED, MDL_NMA_COLLECTIVITY, MDL_NMA_SCORE, MDL_NMA_EIGENVAL)
from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import IntParam, FloatParam, EnumParam
from pyworkflow.utils import *
from pyworkflow.utils.path import makePath, cleanPath, moveFile

from xmipp3 import Plugin
from xmipp3.constants import NMA_HOME
from .convert import getNMAEnviron

NMA_CUTOFF_ABS = 0
NMA_CUTOFF_REL = 1


class FlexProtNMABase(EMProtocol):
    """ Protocol for flexible analysis using NMA. """
    _label = 'nma analysis'

    def _defineParamsCommon(self, form):
        form.addParam('numberOfModes', IntParam, default=20,
                      label='Number of modes',
                      help='The maximum number of modes allowed by the method for '
                           'atomic normal mode analysis is 6 times the number of '
                           'RTB blocks and for pseudoatomic normal mode analysis 3 '
                           'times the number of pseudoatoms. However, the protocol '
                           'allows only up to 200 modes as 20-100 modes are usually '
                           'enough. The number of modes given here should be below '
                           'the minimum between these two numbers.')
        form.addParam('cutoffMode', EnumParam, choices=['absolute', 'relative'],
                      default=NMA_CUTOFF_REL,
                      label='Cut-off mode',
                      help='The cut-off mode can be Absolute or Relative. \n'
                           'Absolute distance allows specifying the maximum distance (in Angstroms) for which it '
                           'is considered that two atoms or pseudoatoms are connected. '
                           'Relative distance allows to specify this distance '
                           'as a percentile of all the distances between '
                           'an atom or a pseudoatom and its nearest neighbors. \n'
                           'For pseudoatoms, the Relative cut-off mode is recommened.')
        form.addParam('rc', FloatParam, default=8,
                      label="Cut-off distance (A)", condition='cutoffMode==%d' % NMA_CUTOFF_ABS,
                      help='Atoms or pseudoatoms beyond this distance will not interact. \n'
                           'For atoms, the distance of 8 Angstroms can work in majority of cases. \n'
                           'For pseudoatoms, it is recommended to use Relative as the cut-off mode, together with '
                           'the Cut-off percentage parameter so that the distance can be computed automatically.')
        form.addParam('rcPercentage', FloatParam, default=95,
                      label="Cut-off percentage", condition='cutoffMode==%d' % NMA_CUTOFF_REL,
                      help='The parameter used to compute the interaction cutoff distance automatically. \n'
                           'The interaction cutoff distance is calculated as the distance below which is '
                           'the percentage of interatomic or interpseudoatomic distances given by this parameter. \n'
                           'Atoms or pseudoatoms beyond the interaction cutoff distance will not interact. \n'
                           'For pseudoatoms, this is the recommended way to compute the interaction cutoff distance, '
                           'obtained via the Relative cut-off mode.')
        form.addParam('collectivityThreshold', FloatParam, default=0.15,
                      label='Threshold on collectivity',
                      help='Collectivity degree is related to the number of atoms or pseudoatoms that are affected by '
                           'the mode, and it is normalized between 0 and 1. Modes below this threshold are deselected in '
                           'the modes metadata file, which means these modes are much less collective. \n'
                           'For no deselection, this parameter should be set to 0 . \n'
                           'Modes 1-6 are always deselected as they are related to rigid-body movements. \n'
                           'The modes metadata file can be used to see which modes are more collective '
                           'in order to decide which modes to use at the image analysis step.')

    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd', 
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'a')
        for l in lines:
            print(fWarn, l)
        fWarn.close()

    def computeModesStep(self, fnPseudoatoms, numberOfModes, cutoffStr):
        (baseDir, fnBase) = os.path.split(fnPseudoatoms)
        fnBase = fnBase.replace(".pdb", "")
        fnDistanceHist = os.path.join(baseDir, 'extra', fnBase + '_distance.hist')
        rc = self._getRc(fnDistanceHist)
        self._enterWorkingDir()
        self.runJob('nma_record_info.py', "%d %s.pdb %d" % (numberOfModes, fnBase, rc), env=getNMAEnviron())
        self.runJob("nma_pdbmat.pl", "pdbmat.dat", env=getNMAEnviron())
        self.runJob("nma_diag_arpack", "", env=getNMAEnviron())
        if not exists("fort.11"):
            self._printWarnings(redStr(
                'Modes cannot be computed. Check the number of '
                'modes you asked to compute and/or consider increasing '
                'cut-off distance. The maximum number of modes allowed by '
                'the method for pseudoatomic normal mode analysis is 3 times '
                'the number of pseudoatoms but the protocol allows only up to '
                '200 modes as 20-100 modes are usually enough.  '
                'If the number of modes is below the minimum between 200 and 3 '
                'times the number of pseudoatoms, consider increasing cut-off distance.'))
        cleanPath("diag_arpack.in", "pdbmat.dat")
        self._leaveWorkingDir()

    def _getRc(self, fnDistanceHist):
        if self.cutoffMode == NMA_CUTOFF_REL:
            rc = self._computeCutoff(fnDistanceHist, self.rcPercentage.get())
        else:
            rc = self.rc.get()
        return rc

    def _computeCutoff(self, fnHist, rcPercentage):
        mdHist = MetaData(fnHist)
        distances = mdHist.getColumnValues(MDL_X)
        distanceCount = mdHist.getColumnValues(MDL_COUNT)
        # compute total number of distances
        nCounts = 0
        for count in distanceCount:
            nCounts += count
        # Compute threshold
        NcountThreshold = nCounts * rcPercentage / 100.0
        nCounts = 0
        for i in range(len(distanceCount)):
            nCounts += distanceCount[i]
            if nCounts > NcountThreshold:
                rc = distances[i]
                break
        msg = "Cut-off distance = %s A" % rc
        print(msg)
        self._enterWorkingDir()
        self._printWarnings(msg)
        self._leaveWorkingDir()

        return rc

    def reformatOutputStep(self, fnPseudoatoms):
        self._enterWorkingDir()
        n = self._countAtoms(fnPseudoatoms)
        self.runJob("nma_reformat_vector_foranimate.pl", "%d fort.11" % n, env=getNMAEnviron())
        self.runJob("cat", "vec.1* > vec_ani.txt")
        self.runJob("rm", "-f vec.1*")
        self.runJob("nma_reformat_vector.pl", "%d fort.11" % n, env=getNMAEnviron())
        fnModesDir = "modes"
        makePath(fnModesDir)
        self.runJob("mv", "-f vec.* %s" % fnModesDir)
        self.runJob("nma_prepare_for_animate.py", "", env=getNMAEnviron())
        self.runJob("rm", "-f vec_ani.txt fort.11 matrice.sdijf")
        moveFile('vec_ani.pkl', 'extra/vec_ani.pkl')
        self._leaveWorkingDir()

    def _countAtoms(self, fnPDB):
        fh = open(fnPDB, 'r')
        n = 0
        for line in fh:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                n += 1
        fh.close()
        return n

    def qualifyModesStep(self, numberOfModes, collectivityThreshold, structureEM, suffix=''):
        self._enterWorkingDir()

        fnVec = glob("modes/vec.*")

        if len(fnVec) < numberOfModes:
            msg = "There are only %d modes instead of %d. "
            msg += "Check the number of modes you asked to compute and/or consider increasing cut-off distance."
            msg += "The maximum number of modes allowed by the method for atomic normal mode analysis is 6 times"
            msg += "the number of RTB blocks and for pseudoatomic normal mode analysis 3 times the number of pseudoatoms. "
            msg += "However, the protocol allows only up to 200 modes as 20-100 modes are usually enough. If the number of"
            msg += "modes is below the minimum between these two numbers, consider increasing cut-off distance."
            self._printWarnings(redStr(msg % (len(fnVec), numberOfModes)))
            print(redStr('Warning: There are only %d modes instead of %d.' % (len(fnVec), numberOfModes)))
            print(redStr("Check the number of modes you asked to compute and/or consider increasing cut-off distance."))
            print(
                redStr("The maximum number of modes allowed by the method for atomic normal mode analysis is 6 times"))
            print(redStr(
                "the number of RTB blocks and for pseudoatomic normal mode analysis 3 times the number of pseudoatoms."))
            print(redStr(
                "However, the protocol allows only up to 200 modes as 20-100 modes are usually enough. If the number of"))
            print(redStr("modes is below the minimum between these two numbers, consider increasing cut-off distance."))

        fnDiag = "diagrtb.eigenfacs"

        if structureEM:
            if which("csh") != "":
                self.runJob("nma_reformatForElNemo.csh", "%d" % len(fnVec), env=getNMAEnviron())
            else:
                if which("bash") != "":
                    self.runJob("nma_reformatForElNemo.sh", "%d" % len(fnVec), env=getNMAEnviron())

            fnDiag = "diag_arpack.eigenfacs"

        self.runJob("echo", "%s | nma_check_modes" % fnDiag, env=getNMAEnviron())
        cleanPath(fnDiag)

        fh = open("Chkmod.res")
        mdOut = MetaData()
        collectivityList = []

        ids, eigvals = self._get_eigval()
        print(eigvals)
        for n in range(len(fnVec)):
            line = fh.readline()
            collectivity = float(line.split()[1])
            collectivityList.append(collectivity)

            objId = mdOut.addObject()
            modefile = self._getPath("modes", "vec.%d" % (n + 1))
            mdOut.setValue(MDL_NMA_MODEFILE, modefile, objId)
            mdOut.setValue(MDL_ORDER, int(n + 1), objId)

            if n >= 6:
                mdOut.setValue(MDL_ENABLED, 1, objId)
            else:
                mdOut.setValue(MDL_ENABLED, -1, objId)
            try:
                mdOut.setValue(MDL_NMA_EIGENVAL, eigvals[n], objId)
            except:
                pass
            mdOut.setValue(MDL_NMA_COLLECTIVITY, collectivity, objId)
            if collectivity < collectivityThreshold:
                mdOut.setValue(MDL_ENABLED, -1, objId)
        fh.close()
        idxSorted = [i[0] for i in sorted(enumerate(collectivityList), key=lambda x: x[1], reverse=True)]

        score = []
        for j in range(len(fnVec)):
            score.append(0)

        modeNum = []
        l = 0
        for k in range(len(fnVec)):
            modeNum.append(k)
            l += 1

        # score = [0]*numberOfModes
        for i in range(len(fnVec)):
            score[idxSorted[i]] = idxSorted[i] + modeNum[i] + 2
        i = 0
        for objId in mdOut:
            score_i = float(score[i]) / (2.0 * l)
            mdOut.setValue(MDL_NMA_SCORE, score_i, objId)
            i += 1
        mdOut.write("modes%s.xmd" % suffix)
        cleanPath("Chkmod.res")

        self._leaveWorkingDir()

    def _get_eigval(self):
        # We are inside the working directory
        fn = 'logs/run.stdout'
        # fn = 'run.stdout'
        content = open(fn, 'r')
        Lines = content.readlines()
        ids = []
        eigval = []
        for line in Lines:
            if line.startswith(' Rdmodfacs> Eigenvector number:'):
                ids.append(int(line[32:]))
            if line.startswith(' Rdmodfacs> Corresponding eigenvalue:'):
                eigval.append(float(line[37:]))
        return ids, eigval

    def _validate(self):
        errors = []
        nmaBin = Plugin.getVar(NMA_HOME)
        nma_programs = ['nma_check_modes',
                        'nma_diag_arpack',
                        'nma_diagrtb',
                        'nma_elnemo_pdbmat']
        # Check Xmipp was compiled with NMA flag to True and
        # some of the nma programs are under the Xmipp/bin/ folder
        for prog in nma_programs:
            if not exists(join(nmaBin, prog)):
                errors.append("Some NMA programs are missing in the NMA folder.")
                # errors.append("Check that Scipion was installed with NMA: 'scipion installb nma'")
                errors.append("Check that Scipion was installed with NMA")
                break
        from pyworkflow.utils.which import which
        if (which("csh") == "") and (which("bash") == ""):
            errors.append("Please install csh (can be a link to tcsh) or bash (e.g., on Ubuntu 'sudo apt-get install "
                          "csh' or 'sudo apt-get install bash')")

        return errors

    def _citations(self):
        return ['harastani2022continuousflex','Nogales2013', 'Jin2014']
