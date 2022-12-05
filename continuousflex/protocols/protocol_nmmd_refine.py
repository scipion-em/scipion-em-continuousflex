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

from continuousflex.protocols.protocol_genesis import *
import pyworkflow.protocol.params as params
from sklearn import decomposition
from xmipp3.convert import writeSetOfVolumes, writeSetOfParticles, readSetOfVolumes, readSetOfParticles
from pwem.constants import ALIGN_PROJ
from continuousflex.protocols.convert import matrix2eulerAngles


class ProtNMMDRefine(ProtGenesis):
    """ Protocol to perform NMMD refinement using GENESIS """
    _label = 'NMMD refine'

    def __init__(self, **kwargs):
        ProtGenesis.__init__(self, **kwargs)
        self._iter = 0
        self._missing_pdbs = None

        # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Refinement')

        form.addParam('numberOfIter', params.IntParam, label="Number of iterations", default=3,
                      help="TODO", important=True)

        form.addParam('numberOfPCA', params.IntParam, label="Number of PCA component", default=5,
                      help="TODO", important=True)

        ProtGenesis._defineParams(self, form)


    def _insertAllSteps(self):
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
                    inp_file = self._getExtraPath("INP_%s" % str(i + 1).zfill(6))
                    outPref = self.getOutputPrefix(i)
                    self._insertFunctionStep("runSimulation", inp_file, outPref)

            self._insertFunctionStep("pdb2dcdStep")

            self._insertFunctionStep("rigidBodyAlignementStep")

            self._insertFunctionStep("updateAlignementStep")

            if self.numberOfIter.get()-1 > iter_global:

                self._insertFunctionStep("newIterationStep")

                self._insertFunctionStep("PCAStep")

                self._insertFunctionStep("runMinimizationStep")


        self._insertFunctionStep("prepareOutputStep")

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
        alignXMD.write(self.getAlignementprefix())

    def updateAlignementStep(self):

        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if self._iter == 0:
                inputSet = self.inputVolume.get()
            else:
                inputSet = self._createSetOfVolumes("inputSet")
                readSetOfVolumes(self.getAlignementprefix(self._iter-1), inputSet)
                inputSet.setSamplingRate(self.inputVolume.get().getSamplingRate())

            inputAlignement = self._createSetOfVolumes("inputAlignement")
            readSetOfVolumes(self.getAlignementprefix(), inputAlignement)
            alignedSet = self._createSetOfVolumes("alignedSet")
        else:
            if self._iter == 0:
                inputSet = self.inputImage.get()
            else:
                inputSet = self._createSetOfParticles("inputSet")
                readSetOfParticles(self.getAlignementprefix(self._iter-1), inputSet)
                inputSet.setSamplingRate(self.inputImage.get().getSamplingRate())

            inputAlignement = self._createSetOfParticles("inputAlignement")
            readSetOfParticles(self.getAlignementprefix(), inputAlignement)
            alignedSet = self._createSetOfParticles("alignedSet")

        alignedSet.setSamplingRate(inputSet.getSamplingRate())
        alignedSet.setAlignment(ALIGN_PROJ)
        iter1 = inputSet.iterItems()
        iter2 = inputAlignement.iterItems()
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

        if isinstance(inputSet, SetOfVolumes):
            writeSetOfVolumes(alignedSet, self.getAlignementprefix())
        else:
            writeSetOfParticles(alignedSet, self.getAlignementprefix())

        self._inputEMMetadata = md.MetaData(self.getAlignementprefix())

    def newIterationStep(self):
        inputPref = self.getInputPDBprefix()
        self._iter += 1
        if self._iter < self.numberOfIter.get():
            inputPref_incr = self.getInputPDBprefix()
            if self.getForceField() == FORCEFIELD_CHARMM:
                runCommand("cp %s.psf %s.psf" % (inputPref, inputPref_incr))
            elif self.getForceField() == FORCEFIELD_CAGO or self.getForceField() == FORCEFIELD_AAGO :
                runCommand("cp %s.top %s.top" % (inputPref, inputPref_incr))

    def PCAStep(self):

        numberOfPCA = self.numberOfPCA.get()

        pdbs_arr = dcd2numpyArr(self._getExtraPath("coords.dcd"))
        nframe, natom,_ = pdbs_arr.shape
        pdbs_matrix = pdbs_arr.reshape(nframe, natom*3)

        pca = decomposition.PCA(n_components=numberOfPCA)
        Y = pca.fit_transform(pdbs_matrix)

        pdb = ContinuousFlexPDBHandler(self.getPDBRef())
        pdb.coords = pca.mean_.reshape(pdbs_matrix.shape[1] // 3, 3)

        matrix = pca.components_.reshape(numberOfPCA,pdbs_matrix.shape[1]//3,3)

        # SAVE NEW inputs
        pdb.write_pdb(self.getInputPDBprefix()+".pdb")
        nm_file = self.getInputPDBprefix()+".nma"
        with open(nm_file, "w") as f:
            for i in range(numberOfPCA):
                f.write(" VECTOR    %i       VALUE  0.0\n" % (i + 1))
                f.write(" -----------------------------------\n")
                for j in range(matrix.shape[1]):
                    f.write(" %e   %e   %e\n" %  (matrix[i,j, 0], matrix[i,j, 1], matrix[i,j, 1]))

    def prepareOutputStep(self):
        for i in range(self.getNumberOfSimulation()):
            outPref = self._getExtraPath("output_%s"% str(i+1).zfill(6))
            cat = "cat "
            for j in range(self.numberOfIter.get()):
                logfile =  self.getOutputPrefix(i,j)+".log"
                if os.path.isfile(logfile):
                    cat += logfile + " "
            runCommand("%s > %s.log"%(cat, outPref))

            dcdfile = self.getOutputPrefix(i,0) + ".dcd"
            if os.path.isfile(dcdfile):
                dcdarr= dcd2numpyArr(dcdfile)
                for j in range(1,self.numberOfIter.get()):
                    dcdfile = self.getOutputPrefix(i,j) + ".dcd"
                    if os.path.isfile(dcdfile):
                        try :
                            dcdarr = np.concatenate((dcdarr, dcd2numpyArr(dcdfile)), axis=0)
                        except ValueError:
                            print("Incomplete DCD file")
                numpyArr2dcd(dcdarr,outPref+ ".dcd")

            pdbfile = self.getOutputPrefix(i)+".pdb"
            if os.path.isfile(pdbfile):
                runCommand("cp %s %s.pdb" % (pdbfile, outPref))

    def runMinimizationStep(self):

        # INP file name
        inp_file = self._getExtraPath("INP_min")
        outPref = self.getInputPDBprefix()+"_min"

        # Inputs files
        args = self.getDefaultArgs()
        args["outputPrefix"]  = outPref
        args["simulationType"]  = SIMULATION_MIN
        args["inputType"]  = INPUT_NEW_SIM
        args["n_steps"]  = 10000
        args["EMfitChoice"]  = EMFIT_NONE

        # Create input genesis file
        createGenesisInput(inp_file, **args)

        # Run minimization
        env = self.getGenesisEnv()
        env.set("OMP_NUM_THREADS", str(self.numberOfThreads.get()))
        runCommand("atdyn %s > %s.log"%(inp_file, outPref), env=env)

        # Copy output pdb
        runCommand("cp %s.pdb %s.pdb"%(outPref, self.getInputPDBprefix()))

    def createGenesisInputStep(self):
        """
        Create GENESIS input files
        :return None:
        """
        for indexFit in range(self.getNumberOfSimulation()):
            inp_file = self._getExtraPath("INP_%s" % str(indexFit + 1).zfill(6))
            args = self.getDefaultArgs(indexFit)
            if self._iter != 0 :
                args["inputType"] = INPUT_NEW_SIM
                args["simulationType"] = SIMULATION_NMMD
                args["nm_number"] =  self.numberOfPCA.get()
                args["nm_dt"] =  0.002
                args["nm_mass"] =  5.0
            createGenesisInput(inp_file, **args)

    def createOutputStep(self):
        ProtGenesis.createOutputStep(self)

    def getPDBRef(self):
        return self._getExtraPath("inputPDB_000001_iter_001.pdb")

    def getInputPDBprefix(self, index=0):
        return ProtGenesis.getInputPDBprefix(self) + "_iter_%s"% str(self._iter+1).zfill(3)

    def getOutputPrefix(self, index=0, itr=None):
        if itr is None : itr = self._iter
        prefix = self._getExtraPath("output_%s_iter_%s"%(
            str(index+1).zfill(6), str(itr+1).zfill(3)))
        return prefix

    def getAlignementprefix(self, itr=None):
        if itr is None : itr = self._iter
        return self._getExtraPath("alignement_iter_%s.xmd"%str(itr+1).zfill(3))

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022continuousflex','vuillemot2022NMMD']

    def _methods(self):
        return []