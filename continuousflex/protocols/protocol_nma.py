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

import os
import math
from os.path import basename, exists, join

from pwem.convert.atom_struct import cifToPdb
from pwem.emlib import MetaData, MDL_NMA_ATOMSHIFT, MDL_NMA_MODEFILE
from pyworkflow.utils import redStr, replaceBaseExt
from pyworkflow.utils.path import copyFile, createLink, makePath, cleanPath, moveFile
from pyworkflow.protocol.params import (PointerParam, IntParam, FloatParam,
                                        LEVEL_ADVANCED)
from pwem.objects import SetOfNormalModes
from xmipp3.base import XmippMdRow
from .protocol_nma_base import FlexProtNMABase, NMA_CUTOFF_REL
from .convert import rowToMode, getNMAEnviron
from pwem import Domain


class FlexProtNMA(FlexProtNMABase):
    """
    Performs Normal Mode Analysis (NMA) on atomic or pseudoatomic
    structural models in order to characterize the intrinsic collective
    motions of biological macromolecules. The protocol generates a set of
    normal modes that can be used for flexibility analysis, conformational
    exploration, structural interpretation, and subsequent image or volume
    analysis workflows.

    AI Generated:

    Normal Mode Analysis (FlexProtNMA) - User Manual
        Overview

        The Normal Mode Analysis protocol is designed to identify and
        characterize the intrinsic motions available to a macromolecular
        structure. Rather than treating a biological assembly as a static
        object, the protocol models its potential collective movements and
        provides a compact description of the directions along which the
        structure can naturally deform.

        In cryo-EM and structural biology studies, many biologically
        important processes involve continuous conformational changes rather
        than discrete structural states. Normal modes offer an efficient way
        to describe these transitions and frequently capture large-scale
        domain rearrangements, hinge motions, and cooperative movements that
        are directly related to molecular function.

        Inputs and Biological Context

        The protocol accepts either an atomic structure or a pseudoatomic
        representation derived from an electron microscopy density map.
        Atomic structures are appropriate when an experimentally determined
        model is available, whereas pseudoatomic models provide a practical
        alternative when only volumetric information exists.

        From a biological perspective, the quality and completeness of the
        input structure strongly influence the relevance of the resulting
        motions. Structures representing functional states, biologically
        meaningful assemblies, or well-resolved density interpretations are
        generally the most suitable starting points for analysis.

        Elastic Network Representation

        The protocol models the structure as an interconnected system whose
        collective movements can be approximated through normal mode theory.
        Interactions are defined according to a selected distance criterion,
        allowing the construction of an elastic representation of the
        molecule.

        Two approaches are available for defining structural connectivity.
        Absolute distance thresholds use a fixed interaction distance,
        whereas relative thresholds determine connectivity from the overall
        distribution of neighboring distances. Relative thresholds are often
        preferred for pseudoatomic models because they adapt more naturally
        to variations in particle density and sampling.

        Choice of Number of Modes

        The protocol computes a user-defined number of normal modes. In most
        biological applications, only a subset of the lowest-frequency
        non-trivial modes is required because these typically correspond to
        the largest and most functionally relevant collective motions.

        Increasing the number of modes may provide a more complete
        description of flexibility, but it also introduces motions that are
        progressively more localized and potentially less biologically
        informative. For many systems, a moderate number of modes is
        sufficient to capture the dominant conformational variability.

        Atomic Structures and RTB Approximation

        When working with atomic models, the protocol uses a coarse-grained
        representation that groups neighboring residues into blocks. This
        strategy enables efficient analysis of large biological assemblies
        while preserving the essential characteristics of collective
        molecular motion.

        The block size influences the balance between computational
        efficiency and structural detail. Larger blocks generally accelerate
        calculations, whereas smaller blocks may provide a more detailed
        description of local flexibility. In most practical situations, the
        default settings offer a suitable compromise.

        Evaluation of Mode Collectivity

        Not all normal modes contribute equally to biologically meaningful
        motions. The protocol evaluates the collectivity of each mode, which
        reflects how broadly a deformation is distributed throughout the
        structure.

        Highly collective modes involve coordinated movement across large
        portions of the molecule and are often associated with functional
        transitions. Less collective modes tend to describe localized
        fluctuations that may be less relevant for global conformational
        analysis. Users can therefore focus subsequent studies on the most
        collective motions.

        Visualization and Animation

        One of the most valuable aspects of normal mode analysis is the
        ability to visualize predicted motions. The protocol generates
        animations that illustrate how the structure deforms along each
        selected mode, helping users interpret the physical meaning of the
        computed motions.

        These animations are intended as qualitative visualizations rather
        than direct representations of experimentally observed amplitudes.
        They provide an intuitive way to identify flexible domains, hinge
        regions, coordinated movements, and potential functional pathways.

        Interpretation of Atomic Displacements

        The protocol also evaluates displacement profiles that indicate how
        strongly different regions of the structure move within each mode.
        These profiles can reveal flexible loops, mobile domains, or regions
        that participate in large conformational transitions.

        From a biological standpoint, regions exhibiting substantial motion
        may correspond to functional interfaces, regulatory elements,
        ligand-binding regions, or structural components involved in
        allosteric communication.

        Outputs and Downstream Applications

        The main output is a set of normal modes associated with the input
        structure. These modes can be used directly for visualization,
        flexibility characterization, conformational interpretation, and
        integration with additional cryo-EM analysis workflows.

        The generated modes frequently serve as the foundation for flexible
        fitting, particle analysis, volume analysis, dimensionality
        reduction, and conformational landscape reconstruction. Because the
        modes provide a compact representation of structural variability,
        they enable efficient exploration of continuous molecular motions.

        Practical Recommendations

        For most biological systems, it is advisable to focus on the
        lowest-frequency collective modes, as these are often the most
        informative and easiest to interpret. Users should visually inspect
        animations and displacement profiles to verify that the predicted
        motions are consistent with known structural and functional
        properties of the molecule.

        When analyzing pseudoatomic models, relative connectivity criteria
        generally provide robust results. For atomic structures, appropriate
        block sizes and realistic interaction parameters help ensure stable
        and biologically meaningful motion predictions.

        Final Perspective

        Normal Mode Analysis provides a powerful bridge between static
        structural models and dynamic biological behavior. By identifying
        the collective motions that a molecule can naturally undergo, the
        protocol offers valuable insight into conformational variability,
        molecular function, and the mechanisms underlying biological
        activity. These modes often form the basis for advanced studies of
        flexibility and continuous heterogeneity in cryo-EM and structural
        biology.
    """
    _label = 'nma analysis'

    def _defineParams(self, form):
        form.addSection(label='Normal Mode Analysis')
        form.addParam('inputStructure', PointerParam, label="Input structure",
                      important=True,
                      pointerClass='AtomStruct',
                      help='The input structure can be an atomic model '
                           '(true PDB) or a pseudoatomic model\n'
                           '(an EM volume converted into pseudoatoms)')
        FlexProtNMABase._defineParamsCommon(self, form)
        form.addParam('rtbBlockSize', IntParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of residues per RTB block (for atomic structures)',
                      help='Used only with atoms. Normal modes of atomic structures are computed with the RTB method. '
                           '\n '
                           'This is the RTB block size. In the RTB method, aminoacids are grouped into blocks of this '
                           'size '
                           'that are moved translationally and rotationally together.')

        form.addSection(label='Animation')
        form.addParam('amplitude', FloatParam, default=50,
                      label='Amplitude',
                      help='Used only for animations of computed normal modes. '
                           'This is the amplitude with which atoms or pseudoatoms are moved '
                           'along normal modes in the animations. \n'
                           'Normal-mode amplitudes corresponding to given images are computed by image analysis.')
        form.addParam('nframes', IntParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Number of frames',
                      help='Number of frames used in animations.')
        form.addParam('downsample', FloatParam, default=1,
                      expertLevel=LEVEL_ADVANCED,
                      # condition=isEm
                      label='Downsample pseudoatoms (for visualization)',
                      help='Used only with pseudoatoms and only for visualization purposes. \n'
                           'A downsample factor of 2 means removing one half of the pseudoatoms.')
        form.addParam('pseudoAtomThreshold', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      # condition=isEm
                      label='Pseudoatom mass threshold (for visualization)',
                      help='Used only with pseudoatoms and only for visualization purposes. \n '
                           'Pseudoatoms whose mass is below this threshold are removed. \n'
                           'The threshold value should be between 0 and 1. '
                           'A threshold of 0 implies no pseudoatom removal.')

    def _insertAllSteps(self):
        # Some steps will differ if the input is a volume or a pdb file
        self.structureEM = self.inputStructure.get().getPseudoAtoms()
        n = self.numberOfModes.get()
        # Link the input
        inputFn = self.inputStructure.get().getFileName()
        localFn = self._getPath(replaceBaseExt(basename(inputFn), 'pdb'))
        self._insertFunctionStep('copyPdbStep', inputFn, localFn,
                                 self.structureEM)

        # Construct string for relative-absolute cutoff
        # This is used to detect when to reexecute a step or not
        cutoffStr = ''
        if self.cutoffMode == NMA_CUTOFF_REL:
            cutoffStr = 'Relative %f' % self.rcPercentage.get()
        else:
            cutoffStr = 'Absolute %f' % self.rc.get()

        # Compute modes
        self.pseudoAtomRadius = 1
        if self.structureEM:
            with open(inputFn, 'r') as fh:
                first_line = fh.readline()
                second_line = fh.readline()
                self.pseudoAtomRadius = float(second_line.split()[2])
            if self.cutoffMode == NMA_CUTOFF_REL:
                params = '-i %s --operation distance_histogram %s' \
                         % (localFn, self._getExtraPath('pseudoatoms_distance.hist'))
                self._insertFunctionStep('analyzePdbStep', params)

            self._insertFunctionStep('computeModesStep', localFn, n, cutoffStr)
            self._insertFunctionStep('reformatOutputStep', "pseudoatoms.pdb")
        else:
            if self.cutoffMode == NMA_CUTOFF_REL:
                params = '-i %s --operation distance_histogram %s' % (
                localFn, self._getExtraPath('atoms_distance.hist'))
                self._insertFunctionStep('analyzePdbStep', params)

            self._insertFunctionStep('computePdbModesStep', n,
                                     self.rtbBlockSize.get(),
                                     cutoffStr)
            self._insertFunctionStep('reformatPdbOutputStep', n)

        self._insertFunctionStep('qualifyModesStep', n,
                                 self.collectivityThreshold.get(),
                                 self.structureEM)
        self._insertFunctionStep('animateModesStep', n,
                                 self.amplitude.get(), self.nframes.get(),
                                 self.downsample.get(),
                                 self.pseudoAtomThreshold.get(),
                                 self.pseudoAtomRadius)
        self._insertFunctionStep('computeAtomShiftsStep', n)
        self._insertFunctionStep('createOutputStep')

    def copyPdbStep(self, inputFn, localFn, isEM):
        """ Copy the input pdb file and also create a link 'atoms.pdb'
        """
        if inputFn.endswith(".cif") or inputFn.endswith(".mmcif"):
            cifToPdb(inputFn, localFn)
        else:
            copyFile(inputFn, localFn)

        if isEM:
            fnOut = self._getPath('pseudoatoms.pdb')
        else:
            fnOut = self._getPath('atoms.pdb')

        if not os.path.exists(fnOut):
            createLink(localFn, fnOut)

        # Keeping only the lines that start with ATOM
        newlines = []
        with open(localFn) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM ") or line.startswith("TER ") or line.startswith("END "):
                newlines.append(line)
        with open(localFn, mode='w') as f:
            f.writelines(newlines)

        # Shifting the atom numbers after line 100000 one step to the left:
        newlines = []
        with open(localFn) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM ") or line.startswith("TER "):
                # print(int(line.split()[1]))
                if int(line.split()[1]) > 99999:
                    if line.startswith("ATOM "):
                        newline = line.replace("ATOM  1", "ATOM 1")
                    else:
                        newline = line.replace("TER   1", "TER  1")
                    newlines.append(newline)
                else:
                    newlines.append(line)
        with open(localFn, mode='w') as f:
            f.writelines(newlines)

    def analyzePdbStep(self, params):
        self.runJob("xmipp_pdb_analysis", params, env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())


    def computePdbModesStep(self, numberOfModes, RTBblockSize, cutoffStr):
        rc = self._getRc(self._getExtraPath('atoms_distance.hist'))

        self._enterWorkingDir()
        # For atoms, the interaction force constant was set to 10 as ElNemo RTB code may ask for its value \
        # (the RTBForceConstant entry was removed from gui as the value does not change the ENM computed normal modes).
        self.runJob('nma_record_info_PDB.py', "%d %d atoms.pdb %f %f"
                    % (numberOfModes, RTBblockSize, rc, 10.0),
                    env=getNMAEnviron())
        self.runJob("nma_elnemo_pdbmat", "", env=getNMAEnviron())
        self.runJob("nma_diagrtb", "", env=getNMAEnviron())

        if not exists("diagrtb.eigenfacs"):
            msg = "Modes cannot be computed. Check the number of modes you " \
                  "asked to compute and/or consider "
            msg += "increasing cut-off distance. The maximum number of " \
                   "modes allowed by the method for atomic "
            msg += "normal mode analysis is 6 times the number of RTB blocks " \
                   "but the protocol allows only up "
            msg += "to 200 modes as 20-100 modes are usually enough. If the " \
                   "number of modes is below the minimum "
            msg += "between 200 and 6 times the number of RTB blocks, consider " \
                   "increasing cut-off distance."
            self._printWarnings(redStr(msg) + '\n')
        self.runJob("rm", "-f *.dat_run diagrtb.dat pdbmat.xyzm pdbmat.sdijf "
                          "pdbmat.dat")

        self._leaveWorkingDir()

    def reformatPdbOutputStep(self, numberOfModes):
        self._enterWorkingDir()

        makePath('modes')
        Natoms = self._countAtoms("atoms.pdb")
        fhIn = open('diagrtb.eigenfacs')
        fhAni = open('vec_ani.txt', 'w')

        for n in range(numberOfModes):
            # Skip two lines
            fhIn.readline()
            fhIn.readline()
            fhOut = open('modes/vec.%d' % (n + 1), 'w')
            for i in range(Natoms):
                line = fhIn.readline()
                fhOut.write(line)
                fhAni.write(line.rstrip().lstrip() + " ")
            fhOut.close()
            if n != (numberOfModes - 1):
                fhAni.write("\n")
        fhIn.close()
        fhAni.close()
        self.runJob("nma_prepare_for_animate.py", "", env=getNMAEnviron())
        cleanPath("vec_ani.txt")
        moveFile('vec_ani.pkl', 'extra/vec_ani.pkl')

        self._leaveWorkingDir()

    def animateModesStep(self, numberOfModes, amplitude, nFrames, downsample,
                         pseudoAtomThreshold, pseudoAtomRadius):
        makePath(self._getExtraPath('animations'))
        self._enterWorkingDir()

        if self.structureEM:
            fn = "pseudoatoms.pdb"
            self.runJob("nma_animate_pseudoatoms.py", "%s extra/vec_ani.pkl 7 %d "
                                                      "%f extra/animations/"
                                                      "animated_mode %d %d %f" % \
                        (fn, numberOfModes, amplitude, nFrames, downsample,
                         pseudoAtomThreshold), env=getNMAEnviron())
        else:
            fn = "atoms.pdb"
            self.runJob("nma_animate_atoms.py", "%s extra/vec_ani.pkl 7 %d %f "
                                                "extra/animations/animated_mode "
                                                "%d" % \
                        (fn, numberOfModes, amplitude, nFrames), env=getNMAEnviron())

        for mode in range(7, numberOfModes + 1):
            fnAnimation = join("extra", "animations", "animated_mode_%03d"
                               % mode)
            fhCmd = open(fnAnimation + ".vmd", 'w')
            fhCmd.write("mol new %s.pdb\n" % self._getPath(fnAnimation))
            fhCmd.write("animate style Loop\n")
            fhCmd.write("display projection Orthographic\n")
            if self.structureEM:
                fhCmd.write("mol modcolor 0 0 Beta\n")
                fhCmd.write("mol modstyle 0 0 Beads %f 8.000000\n"
                            % (pseudoAtomRadius))
            else:
                fhCmd.write("mol modcolor 0 0 Index\n")
                if self._checkPDB_CA(fn):
                    fhCmd.write("mol modstyle 0 0 Beads 1.000000 8.000000\n")
                    # fhCmd.write("mol modstyle 0 0 Beads 1.800000 6.000000 "
                    #         "2.600000 0\n")
                else:
                    fhCmd.write("mol modstyle 0 0 NewRibbons 1.800000 6.000000 "
                                "2.600000 0\n")
            fhCmd.write("animate speed 0.5\n")
            fhCmd.write("animate forward\n")
            fhCmd.close();

        self._leaveWorkingDir()

    def computeAtomShiftsStep(self, numberOfModes):
        fnOutDir = self._getExtraPath("distanceProfiles")
        makePath(fnOutDir)
        maxShift = []
        maxShiftMode = []

        for n in range(7, numberOfModes + 1):
            fnVec = self._getPath("modes", "vec.%d" % n)
            if exists(fnVec):
                fhIn = open(fnVec)
                md = MetaData()
                atomCounter = 0
                for line in fhIn:
                    x, y, z = map(float, line.split())
                    d = math.sqrt(x * x + y * y + z * z)
                    if n == 7:
                        maxShift.append(d)
                        maxShiftMode.append(7)
                    else:
                        if d > maxShift[atomCounter]:
                            maxShift[atomCounter] = d
                            maxShiftMode[atomCounter] = n
                    atomCounter += 1
                    md.setValue(MDL_NMA_ATOMSHIFT, d, md.addObject())
                md.write(join(fnOutDir, "vec%d.xmd" % n))
                fhIn.close()
        md = MetaData()
        for i, _ in enumerate(maxShift):
            fnVec = self._getPath("modes", "vec.%d" % (maxShiftMode[i] + 1))
            if exists(fnVec):
                objId = md.addObject()
                md.setValue(MDL_NMA_ATOMSHIFT, maxShift[i], objId)
                md.setValue(MDL_NMA_MODEFILE, fnVec, objId)
        md.write(self._getExtraPath('maxAtomShifts.xmd'))

    def createOutputStep(self):
        fnSqlite = self._getPath('modes.sqlite')
        nmSet = SetOfNormalModes(filename=fnSqlite)

        md = MetaData(self._getPath('modes.xmd'))
        row = XmippMdRow()

        for objId in md:
            row.readFromMd(md, objId)
            nmSet.append(rowToMode(row))
        inputPdb = self.inputStructure.get()
        nmSet.setPdb(inputPdb)
        self._defineOutputs(outputModes=nmSet)
        self._defineSourceRelation(self.inputStructure, nmSet)

    def _checkPDB_CA(self, fnPDB):
        # This function returns true if all the atoms are CA and P, otherwise false
        from continuousflex.protocols.utilities.pdb_parser import m_inout_read_pdb
        pdb_read = m_inout_read_pdb(fnPDB)
        for atom in pdb_read:
            if atom.type != " C" or atom.loc != "A ":
                if atom.type != " P":
                    return False
        return True

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex']

    def _methods(self):
        pass