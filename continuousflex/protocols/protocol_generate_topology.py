# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
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
# **************************************************************************

from pwem.protocols import EMProtocol
import pyworkflow.protocol.params as params
from pwem.objects.data import AtomStruct
from .utilities.pdb_handler import ContinuousFlexPDBHandler
from pyworkflow.utils import runCommand
import os
from pwem.convert.atom_struct import cifToPdb
import pyworkflow.utils as pwutils
from continuousflex import Plugin
import continuousflex

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2


class ProtGenerateTopology(EMProtocol):
    """
    Generates topology-ready molecular models for GENESIS simulations from
    atomic structures. The protocol prepares biomolecular systems so they can
    be used in molecular dynamics and structure-based modeling workflows.

    AI Generated:

    Generate Topology Model (ProtGenerateTopology) - User Manual
        Overview

        The Generate Topology Model protocol prepares an atomic structure for
        molecular simulations by creating a topology-compatible molecular model.
        Its primary purpose is to transform an experimentally derived structure
        into a form that can be used by simulation engines, including all-atom
        and coarse-grained modeling approaches. This step is often one of the
        first requirements before performing molecular dynamics simulations,
        conformational exploration, or flexible fitting studies.

        The protocol is designed to support different force field strategies.
        Depending on the scientific objective, users may generate a standard
        all-atom representation suitable for detailed molecular dynamics or
        create topology models based on Go-like potentials that emphasize the
        native structure and are commonly used for large-scale conformational
        studies. This flexibility makes the protocol useful for both high-
        resolution atomistic simulations and reduced-complexity models intended
        for studying collective motions.

        Inputs and General Workflow

        The protocol requires an atomic structure as input. The structure may
        originate from experimental methods such as X-ray crystallography,
        cryo-EM, or NMR, as well as from computational modeling procedures.
        Before topology generation, the molecular description is standardized to
        ensure compatibility with the selected simulation framework.

        During preparation, residue and atom naming conventions are harmonized
        and molecular organization is validated. This helps avoid common
        problems arising from differences between structural databases and force
        field expectations. The resulting model is therefore more suitable for
        subsequent simulation steps and less likely to encounter topology-
        generation failures.

        Choice of Force Field Representation

        The protocol supports multiple force field philosophies that address
        different scientific questions.

        The CHARMM option is intended for detailed all-atom simulations where
        atomic interactions are represented explicitly. This approach is
        generally preferred when studying local structural changes, ligand
        interactions, energetic properties, or processes requiring high physical
        realism.

        The All-Atom Go model preserves an atomistic representation while
        simplifying the interaction scheme around the experimentally observed
        native structure. Such models are useful when the goal is to investigate
        large conformational transitions while reducing computational cost.

        The C-Alpha Go model provides an even more reduced representation by
        focusing on backbone-level structural behavior. This option is often
        selected for very large macromolecular assemblies, long-timescale
        simulations, or exploratory studies of conformational landscapes.

        For many biological applications, a topology generated first using an
        all-atom representation can serve as a reliable starting point before
        constructing simplified Go-model variants.

        Residue and Sequence Preparation

        Structural files frequently contain residue numbering irregularities,
        insertion codes, or sequence discontinuities introduced during
        experimental structure determination. These issues may complicate
        topology generation and simulation setup.

        The protocol provides options to reorganize residue numbering and create
        a more consistent molecular description. This is particularly important
        when structures have been assembled from multiple experimental sources
        or contain insertion labels that may not be interpreted consistently by
        simulation software.

        From a biological perspective, correcting residue organization does not
        alter the molecular structure itself but improves the consistency of the
        model used throughout the simulation workflow.

        Protein and Nucleic Acid Systems

        The protocol supports proteins as well as nucleic acid molecules. RNA
        and DNA components are recognized and prepared according to the
        conventions expected by the selected force field representation.

        This capability is important for studies involving ribonucleoprotein
        complexes, chromatin-associated systems, ribosomes, viral genomes, or
        other assemblies containing mixed biomolecular components. By ensuring
        compatibility between molecular components and topology definitions, the
        resulting models are better suited for integrated simulations of complex
        biological systems.

        Structure Standardization

        Experimental structures often contain naming conventions that differ
        from those expected by simulation packages. The protocol performs
        standardization steps that improve compatibility while preserving the
        biological meaning of the model.

        Such preparation is particularly valuable when structures originate from
        different databases, software pipelines, or experimental sources. A
        standardized topology-ready model simplifies downstream simulation
        setup and reduces the need for manual intervention.

        Outputs and Their Interpretation

        The protocol produces a topology-compatible molecular structure ready
        for use in subsequent simulation workflows. The resulting model
        represents the same biological system as the input structure but has
        been adapted to satisfy the requirements of the selected force field and
        simulation environment.

        The output should be viewed as a prepared simulation model rather than a
        modified biological interpretation. Structural features present in the
        original input are preserved while the molecular description is made
        consistent with the computational framework.

        Practical Recommendations

        For detailed molecular dynamics studies, the CHARMM representation is
        generally the preferred choice because it retains the highest level of
        atomic detail. For investigations focused on large conformational
        motions, folding-like transitions, or broad exploration of structural
        landscapes, Go-model representations often provide substantial
        computational advantages.

        Before topology generation, users should inspect the input structure for
        missing regions, unusual residue names, or inconsistencies in sequence
        numbering. Ensuring that the experimental model accurately represents
        the intended biological system will improve the quality of downstream
        simulations.

        When working with mixed protein-nucleic acid assemblies, it is
        particularly important to verify molecular completeness and chain
        organization before beginning the topology preparation process.

        Final Perspective

        Topology generation is a foundational step in molecular simulation
        workflows. Although it is often viewed as a technical preparation stage,
        the quality and consistency of the generated model strongly influence
        the reliability of subsequent analyses. Careful selection of the force
        field representation and thoughtful preparation of the molecular
        structure help ensure that simulation results remain biologically
        meaningful and scientifically robust.
    """
    _label = 'generate topology model'

    def _defineParams(self, form):

        form.addSection(label='Inputs')

        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB",
                      help='Select the input PDB.', important=True)

        form.addParam('forcefield', params.EnumParam, label="Forcefield type", default=FORCEFIELD_CHARMM, important=True,
                       choices=['CHARMM', 'All-atom Go model', 'C-Alpha Go model'],
                       help="Type of the force field used for energy and force calculation. For Go models, it is strongly"
                            " recommended to first generate a topology model using CHARMM, then create a new protocol to generate"
                            " Go model topology based on the output CHARMM all-atom PDB model."
                            " This will ensure that residue sequences are consecutive and TER statements are present in PDB."
                            " CHARMM requires VMD psfgen installed. Go models requires SMOG 2 installed. ")

        form.addParam('reorderResidues', params.BooleanParam, label="Reorder residues and remove insertions", default=False,
                       help='Remove insertion code in the PDB and reorder residues accordingly')

        form.addParam('reorderType', params.BooleanParam, label="Reorder based on segment name?",
                       default=False, condition="reorderResidues",
                       help='If yes reorder the residues within a segment, otherwise, reorder residues within a chain')

    def _insertAllSteps(self):
        ff = self.forcefield.get()

        self._insertFunctionStep("convertInput")

        if ff == FORCEFIELD_CAGO or ff == FORCEFIELD_AAGO:
            self._insertFunctionStep("prepareGROTOP")
            self._insertFunctionStep("runGROTOP")

        if ff == FORCEFIELD_CHARMM:
            self._insertFunctionStep("preparePSF")
            self._insertFunctionStep("runPSF")

        self._insertFunctionStep("checkPDB")
        self._insertFunctionStep("createOutput")

    def convertInput(self):
        inputPDB = self.inputPDB.get().getFileName()
        outPDB = self._getExtraPath("input.pdb")
        ext = os.path.splitext(inputPDB)[1]

        if ext == ".pdb" or ext == ".ent" :
            runCommand("cp %s %s" % (inputPDB, outPDB))
        elif ext == ".cif" or ext == ".mmcif" :
            cifToPdb(inputPDB, outPDB)
        else:
            print("ERROR (toPdb), Unknown file type for file = %s" % inputPDB)

    def createOutput(self):
        self._defineOutputs(outputPDB=AtomStruct(self._getExtraPath("output.pdb")))

    def preparePSF(self):
        inputPDB = self._getExtraPath("input.pdb")
        mol = ContinuousFlexPDBHandler(inputPDB)

        mol.alias_res("HIS", "HSE")
        mol.alias_res("MSE", "MET")
        mol.alias_atom("CD1", "CD", "ILE")

        rna = 0
        rna += mol.alias_res("A", "ADE")
        rna += mol.alias_res("G", "GUA")
        rna += mol.alias_res("C", "CYT")
        rna += mol.alias_res("U", "URA")

        dna = 0
        dna += mol.alias_res("DA", "ADE")
        dna += mol.alias_res("DG", "GUA")
        dna += mol.alias_res("DC", "CYT")
        dna += mol.alias_res("DT", "THY")

        if dna > 0 :
            self.nucleicChoice = NUCLEIC_DNA
        if rna >0 :
            self.nucleicChoice = NUCLEIC_RNA
        else:
            self.nucleicChoice = NUCLEIC_NO


        if self.reorderResidues.get():
            if self.reorderType.get() :
                mol.atom_res_reorder(chainType=1)
            else:
                mol.atom_res_reorder(chainType=0)

        mol.write_pdb(self._getExtraPath("tmp.pdb"))

    def prepareGROTOP(self):
        inputPDB = self._getExtraPath("input.pdb")

        mol = ContinuousFlexPDBHandler(inputPDB)
        # mol.remove_alter_atom()
        mol.remove_hydrogens()
        mol.check_res_order()

        mol.alias_atom("CD", "CD1", "ILE")
        mol.alias_atom("OT1", "O")
        mol.alias_atom("OT2", "OXT")
        mol.alias_res("HSE", "HIS")
        mol.alias_res("HSD", "HIS")
        mol.alias_res("HSP", "HIS")

        rna = 0
        rna += mol.alias_res("CYT", "C")
        rna += mol.alias_res("GUA", "G")
        rna += mol.alias_res("ADE", "A")
        rna += mol.alias_res("URA", "U")

        dna = 0
        dna += mol.alias_res("CYT", "DC")
        dna += mol.alias_res("GUA", "DG")
        dna += mol.alias_res("ADE", "DA")
        dna += mol.alias_res("THY", "DT")
        if dna > 0:
            self.nucleicChoice = NUCLEIC_DNA
        if rna > 0:
            self.nucleicChoice = NUCLEIC_RNA
        else:
            self.nucleicChoice = NUCLEIC_NO

        mol.alias_atom("O1'", "O1*")
        mol.alias_atom("O2'", "O2*")
        mol.alias_atom("O3'", "O3*")
        mol.alias_atom("O4'", "O4*")
        mol.alias_atom("O5'", "O5*")
        mol.alias_atom("C1'", "C1*")
        mol.alias_atom("C2'", "C2*")
        mol.alias_atom("C3'", "C3*")
        mol.alias_atom("C4'", "C4*")
        mol.alias_atom("C5'", "C5*")
        mol.alias_atom("C5M", "C7")
        mol.add_terminal_res()
        if self.reorderResidues.get():
            if self.reorderType.get() :
                mol.atom_res_reorder(chainType=1)
            else:
                mol.atom_res_reorder(chainType=0)
        mol.write_pdb(self._getExtraPath("tmp.pdb"))

    def runPSF(self):
        inputPDB = self._getExtraPath("tmp.pdb")
        inputTopo = self.getCHARMMInputs()[0]
        outputPrefix = self._getExtraPath("output")

        fnPSFgen = self._getExtraPath("psfgen.tcl")
        with open(fnPSFgen, "w") as psfgen:
            psfgen.write("mol load pdb %s\n" % inputPDB)
            psfgen.write("\n")
            psfgen.write("package require psfgen\n")
            psfgen.write("topology %s\n" % inputTopo)
            psfgen.write("\n")
            if self.nucleicChoice == NUCLEIC_RNA or self.nucleicChoice == NUCLEIC_DNA:
                psfgen.write("set nucleic [atomselect top nucleic]\n")
                psfgen.write("set chains [lsort -unique [$nucleic get chain]] ;\n")
                psfgen.write("foreach chain $chains {\n")
                psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
                psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
                psfgen.write("    segment N${chain} { pdb %s_tmp.pdb }\n" % outputPrefix)
                psfgen.write("    coordpdb %s_tmp.pdb N${chain}\n" % outputPrefix)
                if self.nucleicChoice == NUCLEIC_DNA:
                    psfgen.write("    set resids [lsort -unique [$sel get resid]]\n")
                    psfgen.write("    foreach r $resids {\n")
                    psfgen.write("        patch DEOX N${chain}:$r\n")
                    psfgen.write("    }\n")
                psfgen.write("}\n")
                if self.nucleicChoice == NUCLEIC_DNA:
                    psfgen.write("regenerate angles dihedrals\n")
                psfgen.write("\n")
            psfgen.write("set protein [atomselect top protein]\n")
            psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set sel [atomselect top \"protein and pfrag $chain\"]\n")
            psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
            psfgen.write("    segment P${chain} {pdb %s_tmp.pdb}\n" % outputPrefix)
            psfgen.write("    coordpdb %s_tmp.pdb P${chain}\n" % outputPrefix)
            psfgen.write("}\n")
            psfgen.write("rm -f %s_tmp.pdb\n" % outputPrefix)
            psfgen.write("\n")
            psfgen.write("guesscoord\n")
            psfgen.write("writepdb %s.pdb\n" % outputPrefix)
            psfgen.write("writepsf %s.psf\n" % outputPrefix)
            psfgen.write("exit\n")
        fnPSFgen = self._getExtraPath("psfgen.tcl")

        # Run VMD PSFGEN
        from pwem.viewers import Vmd
        runCommand("vmd -dispdev text -e %s" % (fnPSFgen), 
                   env=Vmd.getEnviron())

    def runGROTOP(self):
        outputPrefix = self._getExtraPath("output")
        inputPDB = self._getExtraPath("tmp.pdb")

        # Run Smog2
        environ = pwutils.Environ(os.environ)
        environ.set('PATH', os.path.join(Plugin.getVar("SMOG_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        cmd = "smog2 -i %s -dname %s -%s -limitbondlength -limitcontactlength" %\
                   (inputPDB, outputPrefix,
                    "CA" if self.forcefield.get() == FORCEFIELD_CAGO else "AA")
        runCommand(cmd, env=environ)

        # ADD CHARGE TO TOP FILE
        grotopFile = outputPrefix + ".top"
        with open(grotopFile, 'r') as f1:
            with open(grotopFile + ".tmp", 'w') as f2:
                atom_scope = False
                write_line = False
                for line in f1:
                    if "[" in line and "]" in line:
                        if "atoms" in line:
                            atom_scope = True
                    if atom_scope:
                        if "[" in line and "]" in line:
                            if not "atoms" in line:
                                atom_scope = False
                                write_line = False
                        elif not ";" in line and not (not line or line.isspace()):
                            write_line = True
                        else:
                            write_line = False
                    if write_line:
                        f2.write("%s\t0.0\n" % line[:-1])
                    else:
                        f2.write(line)
        runCommand("cp %s.tmp %s" % (grotopFile, grotopFile))
        runCommand("rm -f %s.tmp" % grotopFile)

        if self.forcefield.get() == FORCEFIELD_CAGO:
            mol = ContinuousFlexPDBHandler(self._getExtraPath("input.pdb"))
            mol.select_atoms(mol.allatoms2ca())
            mol.write_pdb(outputPrefix + ".pdb")
        else:
            runCommand("cp %s %s"%(inputPDB,outputPrefix + ".pdb"))

    def checkPDB(self):
        outPDB = self._getExtraPath("output.pdb")

        # Check PDB
        if not os.path.isfile(outPDB) :
            raise RuntimeError("Can not locate output PDB file %s, check log files for more details " % outPDB)
        if os.path.getsize(outPDB) ==0 :
            raise RuntimeError("PDB file %s is empty, check log files for more details " % outPDB)

        outMol = ContinuousFlexPDBHandler(outPDB)
        if outMol.n_atoms == 0:
            raise RuntimeError("PDB file %s is empty, check log files for more details " % outPDB)


    def getCHARMMInputs(self):
        return continuousflex.__path__[0] + '/protocols/utilities/top_all36_prot_na.rtf',\
         continuousflex.__path__[0] + '/protocols/utilities/par_all36_prot_na.prm',\
         continuousflex.__path__[0] + '/protocols/utilities/toppar_water_ions.str'

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex','vuillemot2022NMMD']

    def _methods(self):
        pass
