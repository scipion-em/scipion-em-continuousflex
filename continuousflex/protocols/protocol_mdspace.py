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
import numpy as np

from continuousflex.protocols.protocol_genesis import *
import pyworkflow.protocol.params as params
from sklearn import decomposition
from xmipp3.convert import writeSetOfVolumes, writeSetOfParticles, readSetOfVolumes, readSetOfParticles
from pwem.constants import ALIGN_PROJ
from continuousflex.protocols.convert import matrix2eulerAngles
from pwem.emlib import MetaData, MDL_ENABLED, MDL_NMA_MODEFILE,MDL_ORDER
from pwem.objects import SetOfNormalModes
from .convert import rowToMode
from xmipp3.base import XmippMdRow

class FlexProtMDSPACE(FlexProtGenesis):

    """ Protocol to perform MDSPACE using GENESIS """
    _label = 'MDSPACE'

    def __init__(self, **kwargs):
        FlexProtGenesis.__init__(self, **kwargs)
        self._iter = 0
        self._missing_pdbs = None

        # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='MDSPACE Refinement')

        form.addParam('numberOfIter', params.IntParam, label="Number of iterations", default=3,
                      help="Number of round of fitting for MDSPACE", important=True)

        form.addParam('numberOfPCA', params.IntParam, label="Number of PCA component", default=5,
                      help="Number of principal component to keep at each round", important=True)

        FlexProtGenesis._defineParams(self, form)


    def _insertAllSteps(self):
        # make path
        self._insertFunctionStep("makePathStep")

        # Convert input PDB
        self._insertFunctionStep("convertInputPDBStep")

        # Convert normal modes
        if (self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD):
            self._insertFunctionStep("convertNormalModeFileStep")

        # Convert input EM data
        if self.EMfitChoice.get() != EMFIT_NONE:
            self._insertFunctionStep("convertInputEMStep")

        for iter_global in range(self.numberOfIter.get()):

            # Create INP files
            self._insertFunctionStep("createGenesisInputStep")

            # RUN simulation
            if not self.disableParallelSim.get() and  \
                self.getNumberOfSimulation() >1  and  existsCommand("parallel") :
                self._insertFunctionStep("runSimulationParallel")
            else:
                if not self.disableParallelSim.get() and  \
                    self.getNumberOfSimulation() >1  and  not existsCommand("parallel"):
                    self.warning("Warning : Can not use parallel computation for GENESIS,"
                                        " please install \"GNU parallel\". Running in linear mode.")
                for i in range(self.getNumberOfSimulation()):
                    inp_file = self.getGenesisInputFile(i)
                    outPref = self.getOutputPrefix(i)
                    self._insertFunctionStep("runSimulation", inp_file, outPref)

            self._insertFunctionStep("pdb2dcdStep")

            self._insertFunctionStep("rigidBodyAlignementStep")

            self._insertFunctionStep("updateAlignementStep")

            self._insertFunctionStep("PCAStep")

            self._insertFunctionStep("runMinimizationStep")

            self._insertFunctionStep("newIterationStep")

        self._insertFunctionStep("createOutputStep")

    def pdb2dcdStep(self):
        pdbs_matrix = []
        missing_pdbs = []

        for i in range(self.getNumberOfSimulation()):
            pdb_fname = self.getOutputPrefix(i) +".pdb"
            if os.path.isfile(pdb_fname) and os.path.getsize(pdb_fname) != 0:
                mol = ContinuousFlexPDBHandler(pdb_fname)
                pdbs_matrix.append(mol.coords)
            else:
                missing_pdbs.append(i)

        pdbs_arr = np.array(pdbs_matrix)

        # save as dcd file
        numpyArr2dcd(pdbs_arr, self._getExtraPath("coords.dcd"))

        # If some pdbs are missing (fitting failed), save indexes
        self._missing_pdbs = np.array(missing_pdbs).astype(int)
        print("MiSSING ARRAY : ")
        print(self._missing_pdbs)

    def rigidBodyAlignementStep(self):

        # open files
        refPDB =  ContinuousFlexPDBHandler(self.getInputPDBprefix()+".pdb")
        arrDCD = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ =arrDCD.shape
        alignXMD = md.MetaData()

        # loop over all pdbs
        for i in range(nframe):
            print("Aligning PDB %i ... " %i)

            # rotate
            coord = arrDCD[i]
            rot_mat, tran = ContinuousFlexPDBHandler.alignCoords(refPDB.coords, coord)
            arrDCD[i] = (np.dot(arrDCD[i], rot_mat) + tran).astype(np.float32)

            # add to MD
            trans_mat = np.zeros((4,4))
            trans_mat[:3,:3] = rot_mat
            trans_mat[:,3][:3] = tran
            rot, tilt, psi,shftx, shfty, shftz = matrix2eulerAngles(trans_mat)
            index = alignXMD.addObject()
            alignXMD.setValue(md.MDL_ANGLE_ROT, rot, index)
            alignXMD.setValue(md.MDL_ANGLE_TILT, tilt, index)
            alignXMD.setValue(md.MDL_ANGLE_PSI, psi, index)
            alignXMD.setValue(md.MDL_SHIFT_X, shftx, index)
            alignXMD.setValue(md.MDL_SHIFT_Y, shfty, index)
            alignXMD.setValue(md.MDL_SHIFT_Z, shftz, index)
            alignXMD.setValue(md.MDL_IMAGE, "", index)

        numpyArr2dcd(arrDCD, self._getExtraPath("coords.dcd"))
        alignXMD.write(self.getTransformation())

    def updateAlignementStep(self):

        if self._iter == 0:
            inputSet = self.inputImage.get()
        else:
            inputSet = self._createSetOfParticles("inputSet")
            readSetOfParticles(self.getAlignementPrefix(self._iter-1), inputSet)
            inputSet.setSamplingRate(self.inputImage.get().getSamplingRate())

        inputTransformation = self._createSetOfParticles("inputTransformation")
        readSetOfParticles(self.getTransformation(), inputTransformation)
        alignedSet = self._createSetOfParticles("alignedSet")

        alignedSet.setSamplingRate(inputSet.getSamplingRate())
        alignedSet.setAlignment(ALIGN_PROJ)
        iter1 = inputSet.iterItems()
        iter2 = inputTransformation.iterItems()
        for i in range(self.getNumberOfSimulation()):
            p1 = iter1.__next__()
            r1 = p1.getTransform()
            if not i in self._missing_pdbs :
                p2 = iter2.__next__()
                r2 = p2.getTransform()
                rot = r2.getRotationMatrix()
                tran = np.array(r2.getShifts()) / inputSet.getSamplingRate()
                new_trans = np.zeros((4, 4))
                new_trans[:3, 3] = tran
                new_trans[:3, :3] = rot
                new_trans[3, 3] = 1.0
                r1.composeTransform(new_trans)
            else:
                print("MISSING PDBS %s"%(i+1))
            p1.setTransform(r1)
            alignedSet.append(p1)

        writeSetOfParticles(alignedSet, self.getAlignementPrefix())
        self._inputEMMetadata = md.MetaData(self.getAlignementPrefix())

    def PCAStep(self):

        numberOfPCA = self.numberOfPCA.get()

        pdbs_arr = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ = pdbs_arr.shape
        pdbs_matrix = pdbs_arr.reshape(nframe, natom*3)

        pca = decomposition.PCA(n_components=numberOfPCA)
        Y = pca.fit_transform(pdbs_matrix)

        pdb = ContinuousFlexPDBHandler(self.getInputPDBprefix()+".pdb")
        pdb.coords = pca.mean_.reshape(pdbs_matrix.shape[1] // 3, 3)

        matrix = pca.components_.reshape(numberOfPCA,pdbs_matrix.shape[1]//3,3)

        # SAVE NEW inputs
        pca_prefix = self.getPCAPrefix()
        pdb.write_pdb(pca_prefix+".pdb")
        nm_file = pca_prefix+".nma"
        np.savetxt(pca_prefix+"_matrix.txt", Y)
        with open(nm_file, "w") as f:
            for i in range(numberOfPCA):
                f.write(" VECTOR    %i       VALUE  0.0\n" % (i + 1))
                f.write(" -----------------------------------\n")
                for j in range(matrix.shape[1]):
                    f.write(" %e   %e   %e\n" %  (matrix[i,j, 0], matrix[i,j, 1], matrix[i,j, 2]))

    def runMinimizationStep(self):

        # INP file name
        inp_file = "%s/INP_min"%self.getGenFiles()

        # copy inputs
        pcapref = self.getPCAPrefix()
        tmppref = self._getExtraPath("tmp")
        runCommand("cp %s.pdb %s.pdb"%(pcapref, tmppref))
        runCommand("cp %s.nma %s.nma"%(pcapref, tmppref))
        if self.getForceField() == FORCEFIELD_CHARMM:
            runCommand("cp %s.psf %s.psf"%(self.getInputPDBprefix(), tmppref))
        else:
            runCommand("cp %s.top %s.top"%(self.getInputPDBprefix(), tmppref))

        # Set Inputs files
        args = self.getDefaultArgs()
        args["inputPDBprefix"]  = tmppref
        args["outputPrefix"]  = self.getInputPDBprefix()+"_min"
        args["simulationType"]  = SIMULATION_MIN
        args["inputType"]  = INPUT_NEW_SIM
        args["n_steps"]  = 10000
        args["EMfitChoice"]  = EMFIT_NONE

        # Create input genesis file
        createGenesisInput(inp_file, **args)

        # Run minimization
        self.runSimulation(inp_file, self.getInputPDBprefix()+"_min")

    def newIterationStep(self):
        for i in range(self.getNumberOfSimulation()):
            outpref = self._getExtraPath("output_%s" % str(i + 1).zfill(6))
            outpref_itr =self.getOutputPrefix(i)
            runCommand("cat %s.log >> %s.log" % (outpref_itr,outpref))

            dcdfile_itr = outpref_itr+ ".dcd"
            if os.path.isfile(dcdfile_itr):
                dcdfile = outpref+".dcd"
                if os.path.isfile(dcdfile):
                    try:
                        dcdarr = np.concatenate((dcd2numpyArr(dcdfile),dcd2numpyArr(dcdfile_itr)), axis=0)
                        numpyArr2dcd(dcdarr, dcdfile)
                    except ValueError:
                        print("Incomplete DCD file")
                else:
                    runCommand("cp %s %s"%(dcdfile_itr, dcdfile))

            pdbfile = outpref_itr+".pdb"
            if os.path.isfile(pdbfile):
                runCommand("cp %s %s.pdb" % (pdbfile, outpref))

        inputPref = self.getInputPDBprefix()
        pcaPref = self.getPCAPrefix()

        ############# INCREMENT ITERATION ###################
        if self._iter < self.numberOfIter.get()-1:
            self._iter += 1
            print("New Iteration %i"%self._iter)
            ###################################################

            inputPref_incr = self.getInputPDBprefix()

            runCommand("cp %s_min.pdb %s.pdb"%(inputPref, inputPref_incr))
            runCommand("cp %s_min.rst %s.rst"%(inputPref, inputPref_incr))
            runCommand("cp %s.nma %s.nma"%(pcaPref, inputPref_incr))
            if self.getForceField() == FORCEFIELD_CHARMM:
                runCommand("cp %s.psf %s.psf" % (inputPref, inputPref_incr))
            elif self.getForceField() == FORCEFIELD_CAGO or self.getForceField() == FORCEFIELD_AAGO :
                runCommand("cp %s.top %s.top" % (inputPref, inputPref_incr))

    def createGenesisInputStep(self):
        """
        Create GENESIS input files
        :return None:
        """
        for indexFit in range(self.getNumberOfSimulation()):
            inp_file = self.getGenesisInputFile(indexFit)
            args = self.getDefaultArgs(indexFit)
            if self._iter != 0 :
                args["inputType"] = INPUT_NEW_SIM
                args["simulationType"] = SIMULATION_NMMD
                args["nm_number"] =  self.numberOfPCA.get()
                args["nm_dt"] =  0.002
                args["nm_mass"] =  5.0
            createGenesisInput(inp_file, **args)

    def createOutputStep(self):
        FlexProtGenesis.createOutputStep(self)

        runCommand("cp %s.pdb %s"%(self.getPCAPrefix(), self.getPath("atoms.pdb")))
        pdb = AtomStruct(self._getPath("atoms.pdb"))
        natoms = ContinuousFlexPDBHandler(self._getPath("atoms.pdb")).n_atoms
        self._defineOutputs(outputMean=pdb)

        makePath(self._getPath("modes"))
        pc_file = self.getPCAPrefix()+".nma"
        with open(pc_file, "r") as f:
            for i in range(self.numberOfPCA.get()):
                f.readline()
                f.readline()
                modefile = self._getPath("modes", "vec.%d" % (i + 1))
                with open(modefile, "w") as fout:
                    for j in range(natoms):
                        fout.write(f.readline())
        mdOut = MetaData()
        for i in range(self.numberOfPCA.get()):
            objId = mdOut.addObject()
            modefile = self._getPath("modes", "vec.%d" % (i + 1))
            mdOut.setValue(MDL_NMA_MODEFILE, modefile, objId)
            mdOut.setValue(MDL_ORDER, i + 1, objId)
            mdOut.setValue(MDL_ENABLED, 1, objId)
        mdOut.write(self._getPath("modes.xmd"))
        pcSet = SetOfNormalModes(filename=self._getPath("modes.sqlite"))
        row = XmippMdRow()
        for objId in mdOut:
            row.readFromMd(mdOut, objId)
            pcSet.append(rowToMode(row))
        pcSet.setPdb(pdb)
        self._defineOutputs(outputPCA=pcSet)

    def getOutputPrefix(self, index=0):
        return self._getExtraPath("output_%s_iter_%s" % (str(index + 1).zfill(6),str(self._iter+1).zfill(3)))

    def getInputPDBprefix(self, index=0):
        """
        Get the input PDB prefix of the specified index
        :param int index: index of input PDB
        :return str: Input PDB prefix
        """
        prefix = self._getExtraPath("inputPDB_%s_iter_%s")
        if self.getNumberOfInputPDB() == 1:
            return prefix % (str(1).zfill(6),str(self._iter+1).zfill(3))
        else:
            return prefix % (str(index + 1).zfill(6),str(self._iter+1).zfill(3))
    def getPCAPrefix(self):
        return self._getExtraPath("pca_iter_%s" % (str(self._iter+1).zfill(3)))

    def getAlignementPrefix(self, itr=None):
        if itr is None : itr = self._iter
        return "%s/alignement_iter_%s.xmd"%(self.getEmdFiles(),str(itr+1).zfill(3))
    def getTransformation(self, itr=None):
        if itr is None : itr = self._iter
        return "%s/transformation_iter_%s.xmd"%(self.getEmdFiles(),str(itr+1).zfill(3))

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['vuillemot2023mdspace','vuillemot2022NMMD','harastani2022continuousflex']

    def _methods(self):
        return []