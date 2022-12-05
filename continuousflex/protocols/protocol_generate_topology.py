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


NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2


class ProtGenerateTopology(EMProtocol):
    """ Protocol to generate topology files for GENESIS simulations """
    _label = 'generate topology'

    def _defineParams(self, form):

        form.addSection(label='Inputs')

        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB",
                      help='Select the input PDB.', important=True)

        group = form.addGroup('Forcefield Inputs')
        group.addParam('forcefield', params.EnumParam, label="Forcefield type", default=FORCEFIELD_CHARMM, important=True,
                       choices=['CHARMM', 'All-atom Go model', 'C-Alpha Go model'],
                       help="Type of the force field used for energy and force calculation. For Go models, it is strongly"
                            " recommended to first generate topology using CHARMM, then create a new protocol to generate"
                            " Go model topology based on the output CHARMM all-atom PDB model."
                            " This will ensure that residue sequences are consecutive and TER statements are present in PDB."
                            " CHARMM requires VMD psfgen installed. Go models requires SMOG 2 installed. ")
        group.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=NUCLEIC_NO,
                       choices=['No', 'RNA', 'DNA'], help="Specify if the generator should consider nucleic residues as DNA or RNA")

        group.addParam('inputPRM', params.FileParam, label="CHARMM parameter file (prm)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM,
                       help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')
        group.addParam('inputRTF', params.FileParam, label="CHARMM topology file (rtf)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM,
                       help='CHARMM topology file containing information about atom connectivity of residues and'
                            ' other molecules. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')
        group.addParam('inputSTR', params.FileParam, label="CHARMM stream file (str, optional)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM, default="",
                       help='CHARMM stream file containing both topology information and parameters. '
                            'Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')

        group.addParam('smog_dir', params.FileParam, label="SMOG 2 install directory",
                       help="Path to SMOG2 install directory (For SMOG2 installation, see "
                            "https://smog-server.org/smog2/). If SMOG2 is not installed, you can use the web GUI instead "
                            "https://smog-server.org/cgi-bin/GenTopGro.pl (Recommended to run the protocol with empty smog_dir,"
                            " when the protocol fails, get the input.pdb file generate in the extra directory as input of SMOG server)",
                       condition="(forcefield==%i or forcefield==%i)"%(FORCEFIELD_CAGO, FORCEFIELD_AAGO))

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
        inputTopo = self.inputRTF.get()
        outputPrefix = self._getExtraPath("output")
        nucleicChoice = self.nucleicChoice.get()

        fnPSFgen = self._getExtraPath("psfgen.tcl")
        with open(fnPSFgen, "w") as psfgen:
            psfgen.write("mol load pdb %s\n" % inputPDB)
            psfgen.write("\n")
            psfgen.write("package require psfgen\n")
            psfgen.write("topology %s\n" % inputTopo)
            psfgen.write("pdbalias residue HIS HSE\n")
            psfgen.write("pdbalias residue MSE MET\n")
            psfgen.write("pdbalias atom ILE CD1 CD\n")
            if nucleicChoice == NUCLEIC_RNA:
                psfgen.write("pdbalias residue A ADE\n")
                psfgen.write("pdbalias residue G GUA\n")
                psfgen.write("pdbalias residue C CYT\n")
                psfgen.write("pdbalias residue U URA\n")
            elif nucleicChoice == NUCLEIC_DNA:
                psfgen.write("pdbalias residue DA ADE\n")
                psfgen.write("pdbalias residue DG GUA\n")
                psfgen.write("pdbalias residue DC CYT\n")
                psfgen.write("pdbalias residue DT THY\n")
            psfgen.write("\n")
            if nucleicChoice == NUCLEIC_RNA or nucleicChoice == NUCLEIC_DNA:
                psfgen.write("set nucleic [atomselect top nucleic]\n")
                psfgen.write("set chains [lsort -unique [$nucleic get chain]] ;\n")
                psfgen.write("foreach chain $chains {\n")
                psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
                psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
                psfgen.write("    segment N${chain} { pdb %s_tmp.pdb }\n" % outputPrefix)
                psfgen.write("    coordpdb %s_tmp.pdb N${chain}\n" % outputPrefix)
                if nucleicChoice == NUCLEIC_DNA:
                    psfgen.write("    set resids [lsort -unique [$sel get resid]]\n")
                    psfgen.write("    foreach r $resids {\n")
                    psfgen.write("        patch DEOX N${chain}:$r\n")
                    psfgen.write("    }\n")
                psfgen.write("}\n")
                if nucleicChoice == NUCLEIC_DNA:
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

    def runPSF(self):
        fnPSFgen = self._getExtraPath("psfgen.tcl")
        outputPrefix = self._getExtraPath("output")

        # Run VMD PSFGEN
        runCommand("vmd -dispdev text -e %s > %s.log " % (fnPSFgen, outputPrefix))


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

        if self.nucleicChoice.get() == NUCLEIC_RNA:
            mol.alias_res("CYT", "C")
            mol.alias_res("GUA", "G")
            mol.alias_res("ADE", "A")
            mol.alias_res("URA", "U")

        elif self.nucleicChoice.get() == NUCLEIC_DNA:
            mol.alias_res("CYT", "DC")
            mol.alias_res("GUA", "DG")
            mol.alias_res("ADE", "DA")
            mol.alias_res("THY", "DT")

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
        mol.atom_res_reorder()
        mol.write_pdb(inputPDB)

    def runGROTOP(self):
        outputPrefix = self._getExtraPath("output")
        inputPDB = self._getExtraPath("input.pdb")

        # Run Smog2
        runCommand("%s/bin/smog2" % self.smog_dir.get() + \
                   " -i %s -dname %s -%s -limitbondlength -limitcontactlength > %s.log" %
                   (inputPDB, outputPrefix,
                    "CA" if self.forcefield.get() == FORCEFIELD_CAGO else "AA", outputPrefix))

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
            mol = ContinuousFlexPDBHandler(inputPDB)
            mol.select_atoms(mol.allatoms2ca())
            mol.write_pdb(outputPrefix + ".pdb")
        else:
            runCommand("cp %s %s"%(inputPDB,outputPrefix + ".pdb"))

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['harastani2022continuousflex','vuillemot2022NMMD']

    def _methods(self):
        pass
