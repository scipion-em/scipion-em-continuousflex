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


from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
import pyworkflow.protocol.params as params
from continuousflex.protocols.protocol_genesis import ProtGenesis
from .plotter import FlexPlotter
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
from xmippLib import SymList
import pwem.emlib.metadata as md


from continuousflex.protocols.utilities.genesis_utilities import PDBMol, matchPDBatoms,compute_pca

class GenesisViewer(ProtocolViewer):
    """ Visualization of results from the GENESIS protocol
    """
    _label = 'viewer genesis'
    _targets = [ProtGenesis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('fitRange', params.NumericRangeParam,
                      label="List of fitting to display",
                      default='1',
                      help=' Examples:\n'
                           ' "1,3-5" -> [1,3,4,5]\n'
                           ' "1, 2, 4" -> [1,2,4]\n')
        form.addParam('displayEnergy', params.LabelParam,
                      label='Display Energy',
                      help='TODO')

        form.addParam('displayCC', params.LabelParam,
                      label='Display correlation coefficient',
                      help='TODO')

        form.addParam('displayRMSD', params.LabelParam,
                      label='Display RMSD',
                      help='TODO')

        form.addParam('targetPDB', params.PathParam, default=None,
                        label="List of Target PDBs",
                        help='Use the file pattern as file location with /*.pdb')

        form.addParam('displayAngularDistance', params.LabelParam,
                      label='Display Angular distance',
                      help='TODO')

        form.addParam('rigidBodyParams', params.FileParam, default=None,
                        label="Target Rigid Body Parameters",
                        help='TODO')


        form.addParam('displayPCA', params.LabelParam,
                      label='Display PCA',
                      help='TODO')

    def _getVisualizeDict(self):
        return {
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayRMSD': self._plotRMSD,
            'displayAngularDistance': self._plotAngularDistance,
            'displayPCA': self._plotPCA,
                }


    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "CC")
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            log_file = readLogFile(self.protocol._getExtraPath("%s_output.log" % (str(i).zfill(5))))
            for e in ene_default:
                if e in log_file:
                    if e in ene :
                        ene[e].append(log_file[e])
                    else:
                        ene[e] = [log_file[e]]

        x = self.getStep(log_file["STEP"])

        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7, errorevery=len(log_file["STEP"]) //10)
        plotter.legend()
        plotter.show()

    def _plotEnergyDetail(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "CC")
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            log_file = readLogFile(self.protocol._getExtraPath("%s_output.log" % (str(i).zfill(5))))
            for e in ene_default:
                if e in log_file:
                    if e in ene :
                        ene[e].append(log_file[e])
                    else:
                        ene[e] = [log_file[e]]

        x = self.getStep(log_file["STEP"])

        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7, errorevery=len(log_file["STEP"]) //10)
        plotter.legend()
        plotter.show()

    def _plotCC(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Correlation coefficient", "Time (ps)", "CC")

        # Get CC list
        fitlist = self.getFitlist()
        cc = []
        for i in fitlist:
            outputPrefix = self.protocol._getExtraPath("%s_output" % (str(i).zfill(5)))
            log_file = readLogFile(outputPrefix + ".log")
            cc.append(log_file['RESTR_CVS001'])

        # Plot CC
        x = self.getStep(log_file["STEP"])
        for i in range(len(cc)):
            if len(cc) <= 10:
                ax.plot(x, cc[i], color="tab:blue", alpha=0.3)
        ax.errorbar(x = x, y=np.mean(cc, axis=0), yerr=np.std(cc, axis=0),
                    capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", errorevery=len(log_file["STEP"]) //10)

        plotter.show()

    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "Time (ps)", "RMSD ($\AA$)")
        target_pdbs_list = self.getTargetList()

        # Get RMSD list
        fitlist = self.getFitlist()
        print("////")
        print(fitlist)
        print(target_pdbs_list)
        rmsd = []
        for i in range(len(fitlist)):
            outputPrefix = self.protocol._getExtraPath("%s_output" % (str(i+1).zfill(5)))
            log_file = readLogFile(outputPrefix + ".log")
            rmsd.append(rmsdFromDCD(outputPrefix=outputPrefix, inputPDB=self.protocol.getInputPDBprefix(i)+".pdb",
                    targetPDB=target_pdbs_list[0] if len(target_pdbs_list)  == 1 else target_pdbs_list[i]))

        # Plot RMSD
        for i in range(len(rmsd)):
            x = self.getStep(log_file["STEP"])[:len(rmsd[i])]
            if len(rmsd) <=10:
                ax.plot(x, rmsd[i], color="tab:blue", alpha=0.3)

        ax.errorbar(x = x, y=np.mean(rmsd, axis=0), yerr=np.std(rmsd, axis=0),
                    capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", errorevery=len(log_file["STEP"]) //10)

        plotter.show()


    def getFitlist(self):
        return np.array(getListFromRangeString(self.fitRange.get()))

    def _plotAngularDistance(self, paramName):
        angular_dist = []
        shift_dist = []
        mdImgGT = md.MetaData(self.rigidBodyParams.get())
        fitlist = self.getFitlist()
        for i in fitlist:
            rot0 = mdImgGT.getValue(md.MDL_ANGLE_ROT, int(i))
            tilt0 = mdImgGT.getValue(md.MDL_ANGLE_TILT, int(i))
            psi0 = mdImgGT.getValue(md.MDL_ANGLE_PSI, int(i))
            shiftx0 = -mdImgGT.getValue(md.MDL_SHIFT_X, int(i))
            shifty0 = -mdImgGT.getValue(md.MDL_SHIFT_Y, int(i))

            mdImgFn = self.protocol._getExtraPath("%s_current_angles.xmd" % (str(i).zfill(5)))
            mdImg = md.MetaData(mdImgFn)
            rot = mdImg.getValue(md.MDL_ANGLE_ROT, 1)
            tilt = mdImg.getValue(md.MDL_ANGLE_TILT, 1)
            psi = mdImg.getValue(md.MDL_ANGLE_PSI, 1)
            shiftx = -mdImg.getValue(md.MDL_SHIFT_X, 1)
            shifty = -mdImg.getValue(md.MDL_SHIFT_Y, 1)

            angular_dist.append(SymList.computeDistanceAngles(SymList(),
                    rot, tilt, psi, rot0, tilt0, psi0, False, True, False))

            shift_dist.append(np.linalg.norm(np.array([shiftx, shifty, 0.0])
                                             - np.array([shiftx0, shifty0, 0.0])))

        plotter1 = FlexPlotter()
        ax1 = plotter1.createSubPlot("Angular Distance (°)", "# Image", "Angular Distance (°)")
        ax1.plot(angular_dist, "o")
        plotter1.show()

        plotter2 = FlexPlotter()
        ax2 = plotter2.createSubPlot("Shift Distance ($\AA$)", "# Image", "Shift Distance ($\AA$)")
        ax2.plot(shift_dist, "o")
        plotter2.show()

    def _plotPCA(self, paramName):

        initPDB = PDBMol(self.protocol.getInputPDBprefix(0)+".pdb")

        # MAtch atoms with target
        if self.targetPDB.get() is not None:
            targetPDBlist = self.getTargetList()
            targetPDB = PDBMol(targetPDBlist[0])
            idx = matchPDBatoms([initPDB,targetPDB], ca_only=False)
        else:
            idx = np.array([np.arange(initPDB.n_atoms)]).T

        # Get Init PDB coords
        initPDBs = []
        for i in range(self.protocol.getNumberOfInputPDB()):
            mol = PDBMol(self.protocol.getInputPDBprefix(i)+".pdb")
            initPDBs.append(mol.coords[idx[:,0]].flatten())

        # Get fitted PDBs coords
        fitlist = self.getFitlist()
        fitPDBs = []
        for i in fitlist:
            mol = PDBMol(self.protocol._getExtraPath("%s_output.pdb" % (str(i).zfill(5))))
            fitPDBs.append(mol.coords[idx[:,0]].flatten())

        data = fitPDBs + initPDBs
        length=[len(fitPDBs), len(initPDBs)]
        labels=["Fitted PDBs", "Init. PDBs"]

        # Get TargetPDBs coords
        if self.targetPDB.get() is not None:
            targetPDBlist = self.getTargetList()
            targetPDBs=[]
            for i in targetPDBlist:
                targetPDBs.append(PDBMol(i).coords[idx[:,1]].flatten())
            data = data+targetPDBs
            length.append(len(targetPDBs))
            labels.append("Target PDBs")

        # Display PCA
        initPDB.select_atoms(idx[:,0])
        fig, ax=compute_pca(data=data, length=length, labels=labels,
                    n_components=2, figsize=(5, 5), initdcd=initPDB)
        fig.show()

    def getStep(self, step):
        time_step = float( self.protocol.time_step.get())
        return np.arange(len(step))*(step[1]-step[0]) * time_step

    def getTargetList(self):
        targetPDBlistall = [f for f in glob.glob(self.targetPDB.get())]
        targetPDBlistall.sort()
        targetPDBlist = []
        for i in self.getFitlist():
            targetPDBlist.append(targetPDBlistall[i-1])
        return targetPDBlist




def readLogFile(log_file):
    with open(log_file,"r") as file:
        header = None
        dic = {}
        for line in file:
            if line.startswith("INFO:"):
                if header is None:
                    header = line.split()
                    for i in range(1,len(header)):
                        dic[header[i]] = []
                else:
                    splitline = line.split()
                    if len(splitline) == len(header):
                        for i in range(1,len(header)):
                            try :
                                dic[header[i]].append(float(splitline[i]))
                            except ValueError:
                                pass

    return dic

def rmsdFromDCD(outputPrefix, inputPDB, targetPDB):

    # EXTRACT PDBs from dcd file
    with open("%s_tmp_dcd2pdb.tcl" % outputPrefix, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s.dcd\n" % (inputPDB, outputPrefix)
        s += "set nf [molinfo top get numframes]\n"
        s += "for {set i 0 } {$i < $nf} {incr i} {\n"
        s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" % outputPrefix
        s += "}\n"
        s += "exit\n"
        f.write(s)
    os.system("vmd -dispdev text -e %s_tmp_dcd2pdb.tcl > /dev/null" % outputPrefix)

    # DEF RMSD
    def RMSD(c1, c2):
        return np.sqrt(np.mean(np.square(np.linalg.norm(c1 - c2, axis=1))))

    # COMPUTE RMSD
    rmsd = []
    inputPDBmol = PDBMol(inputPDB)
    targetPDBmol = PDBMol(targetPDB)

    idx = matchPDBatoms([inputPDBmol, targetPDBmol], ca_only=True)
    rmsd.append(RMSD(inputPDBmol.coords[idx[:, 0]], targetPDBmol.coords[idx[:, 1]]))
    i=0
    while(os.path.exists("%stmp%i.pdb"%(outputPrefix,i+1))):
        f = "%stmp%i.pdb"%(outputPrefix,i+1)
        mol = PDBMol(f)
        rmsd.append(RMSD(mol.coords[idx[:, 0]], targetPDBmol.coords[idx[:, 1]]))
        i+=1

    # CLEAN TMP FILES AND SAVE
    os.system("rm -f %stmp*" % (outputPrefix))
    return rmsd