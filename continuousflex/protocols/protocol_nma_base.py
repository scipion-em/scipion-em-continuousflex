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
    """
    Provides the foundational framework for Normal Mode Analysis (NMA) of
    macromolecular structures and pseudoatomic models. The protocol is intended
    to characterize intrinsic molecular flexibility by identifying collective
    motions that describe biologically relevant conformational changes. These
    motions can later be used in flexible fitting, variability analysis, motion
    exploration, and the interpretation of structural heterogeneity observed in
    cryo-EM experiments.

    AI Generated:

    Normal Mode Analysis Base (FlexProtNMABase) - User Manual
        Overview

        The Normal Mode Analysis Base protocol provides the core functionality
        required to compute and evaluate normal modes describing the intrinsic
        flexibility of molecular structures. In structural biology, normal mode
        analysis is widely used to investigate how large biomolecular assemblies
        move between different conformational states while preserving their
        overall architecture.

        The protocol is designed to support both atomic and pseudoatomic
        representations. This flexibility allows researchers to study systems
        ranging from high-resolution atomic models to lower-resolution cryo-EM
        reconstructions that have been converted into pseudoatomic forms. The
        resulting modes provide a compact description of collective molecular
        motions and often reveal biologically meaningful pathways of structural
        change.

        Biological Significance of Normal Modes

        Biological macromolecules are dynamic entities rather than static
        structures. Proteins, ribosomes, viral capsids, and molecular machines
        frequently perform their functions through coordinated movements of
        domains, subunits, or flexible regions. Normal mode analysis seeks to
        identify these collective motions and rank them according to their
        energetic accessibility.

        Lower-frequency modes are often the most biologically relevant because
        they describe large-scale coordinated movements that can be associated
        with ligand binding, allosteric regulation, assembly rearrangements,
        transport mechanisms, or transitions between functional states.
        Understanding these motions can provide valuable insight into molecular
        mechanisms that are difficult to infer from a single static structure.

        Defining the Elastic Network

        A key aspect of normal mode analysis is the definition of interactions
        between atoms or pseudoatoms. The protocol allows users to control how
        neighboring elements are connected through an interaction cutoff.

        For atomic structures, a fixed interaction distance is often suitable
        because atomic coordinates provide detailed geometric information. For
        pseudoatomic models derived from cryo-EM maps, relative cutoffs are
        generally preferred because they adapt automatically to the density and
        distribution of pseudoatoms. This often produces more stable and
        physically meaningful elastic networks.

        Choosing an appropriate interaction range is important because it
        determines the balance between local rigidity and global flexibility.
        Cutoffs that are too restrictive may fragment the network and prevent
        meaningful mode calculation, whereas excessively large cutoffs may
        suppress biologically relevant flexibility.

        Number of Modes

        The protocol allows users to select how many normal modes should be
        computed. In most biological applications, only a subset of the
        available modes is required because the lowest-frequency collective
        motions typically capture the most relevant conformational variability.

        A moderate number of modes is often sufficient for exploring structural
        flexibility, generating deformed models, or performing downstream
        conformational analyses. Computing an excessive number of modes may
        increase computational cost without providing additional biological
        insight.

        Mode Qualification and Selection

        Not all computed modes are equally informative. The protocol evaluates
        the collective nature of each mode and identifies those that are most
        likely to represent meaningful concerted motions.

        Collectivity measures the extent to which a motion involves large
        portions of the structure rather than only a few localized elements.
        Highly collective modes are often associated with functional molecular
        rearrangements, whereas poorly collective modes may correspond to local
        fluctuations with limited biological significance.

        The protocol therefore provides mechanisms for identifying and
        prioritizing the most informative modes. This helps users focus on
        motions that are more likely to contribute to biologically relevant
        conformational transitions.

        Interpretation of Eigenvalues and Flexibility

        Each normal mode is associated with an eigenvalue that reflects the
        energetic cost of the corresponding motion. Lower eigenvalues indicate
        softer motions that can occur more easily, while higher eigenvalues
        correspond to increasingly constrained deformations.

        From a biological perspective, the lowest-frequency non-rigid-body modes
        are often the most informative because they represent motions that the
        molecular system can naturally access. These modes frequently correlate
        with experimentally observed conformational variability.

        Outputs and Their Interpretation

        The protocol produces a collection of normal modes together with
        quantitative descriptors that help evaluate their importance. The
        resulting mode set can be used directly in downstream flexibility
        analyses, conformational sampling, flexible fitting procedures, and
        structural interpretation workflows.

        Researchers can inspect the relative importance of different modes,
        evaluate their collectivity, and determine which motions should be used
        for subsequent analyses. The outputs provide a structured description of
        the accessible conformational space surrounding the input structure.

        Practical Recommendations

        For pseudoatomic models derived from cryo-EM maps, relative interaction
        cutoffs are generally recommended because they adapt more naturally to
        the pseudoatom distribution. For atomic structures, carefully chosen
        absolute cutoffs often provide reliable results.

        In most studies, attention should focus on the lowest-frequency
        collective modes rather than attempting to interpret every computed
        motion. Reviewing collectivity values and ensuring that the interaction
        network is sufficiently connected are important quality-control steps.

        When mode computation becomes unstable or produces fewer modes than
        expected, increasing the interaction cutoff is often an effective way to
        improve network connectivity and obtain a more complete description of
        molecular flexibility.

        Final Perspective

        Normal mode analysis provides a powerful bridge between static
        structural models and the dynamic behavior of biological molecules. By
        identifying collective motions that are energetically accessible, the
        protocol enables researchers to explore conformational landscapes,
        interpret experimental heterogeneity, and gain mechanistic insight into
        the functional flexibility of complex macromolecular systems.
    """
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
