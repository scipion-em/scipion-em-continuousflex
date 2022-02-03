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
from continuousflex.protocols.protocol_genesis import *
from continuousflex.protocols.utilities.genesis_utilities import *

from .plotter import FlexPlotter
from pwem.viewers import VmdView
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
import pwem.emlib.metadata as md

from sklearn.decomposition import PCA


class GenesisViewer(ProtocolViewer):
    """ Visualization of results from the GENESIS protocol
    """
    _label = 'viewer genesis'
    _targets = [ProtGenesis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('fitRange', params.NumericRangeParam,
                      label="Simulation number",
                      default='1',important = True,
                      help=' The simulation numbers to display. Examples:'
                           ' "1,3-5" -> [1,3,4,5]'
                           ' "1, 2, 4" -> [1,2,4]')
        group = form.addGroup('Energy Analysis')
        group.addParam('displayEnergy', params.LabelParam,
                      label='Display Potential Energy',
                      help='Show time series of the potentials used in MD simulation/Minimization')

        group = form.addGroup('RMSD analysis')
        group.addParam('targetPDB', params.PathParam, default=None,
                        label="Target PDB (s)", important=True,
                        help='Use the file pattern as file location with /*.pdb')
        group.addParam('referencePDB', params.PathParam, default="",
                        label="Reference PDB (optional)",
                        help='TODO')

        group.addParam('displayRMSDts', params.LabelParam,
                      label='Display RMSD time series',
                      help='TODO')

        group.addParam('displayRMSD', params.LabelParam,
                      label='Display RMSD',
                      help='TODO')

        group.addParam('alignTarget', params.BooleanParam, default=False,
                        label="Align Target PDB",
                        help='TODO')

        if self.protocol.EMfitChoice.get() != EMFIT_NONE:
            group = form.addGroup('Cryo EM fitting')
            group.addParam('displayCC', params.LabelParam,
                          label='Display Correlation Coefficient',
                          help='Show C.C. time series during the simulation')

        group = form.addGroup('PCA analysis')
        group.addParam('displayPCA', params.LabelParam,
                      label='Display PCA',
                      help='TODO')

        group = form.addGroup('Simulation trajectory')
        group.addParam('displayTrajVMD', params.LabelParam,
                      label='Display Trajecory',
                      help='TODO')

    def _getVisualizeDict(self):
        return {
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayRMSDts': self._plotRMSDts,
            'displayRMSD': self._plotRMSD,
            'displayPCA': self._plotPCA,
            'displayTrajVMD': self._plotTrajVMD,
                }

    def _plotTrajVMD(self, paramName):
        fitlist = self.getFitlist()
        vmdviewer = VmdView("%s.pdb %s.dcd"%(self.protocol.getInputPDBprefix(fitlist[0] - 1),
                    self.protocol.getOutputPrefixAll(fitlist[0] - 1)[0]))
        vmdviewer.show()

    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "Energy")
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]

        x = np.arange(len(log_file["TOTAL_ENE"]))*\
            (int(self.protocol.eneout_period.get()) )*\
            float( self.protocol.time_step.get())
        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7,
                        errorevery=np.max([len(log_file["STEP"]) //10,1]))
        plotter.legend()
        plotter.show()

    def _plotEnergyDetail(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "Energy")
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT", "RESTRAINT_TOTAL"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                log_file = readLogFile(j+".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]

        x = np.arange(len(log_file["BOND"])) * \
            (int(self.protocol.eneout_period.get())) * \
            float(self.protocol.time_step.get())
        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7,
                        errorevery=np.max([len(log_file["STEP"]) //10,1]))
        plotter.legend()
        plotter.show()

    def _plotCC(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Correlation coefficient", "Time (ps)", "CC")

        # Get CC list
        fitlist = self.getFitlist()
        cc = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i-1)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                cc.append(log_file['RESTR_CVS001'])

        # Plot CC
        for i in range(len(cc)):
            x = np.arange(len(cc[i])) * \
                (int(self.protocol.eneout_period.get())) * \
                float(self.protocol.time_step.get())
            if len(cc) <= 50:
                ax.plot(x, cc[i], alpha=0.3, label="#%i"%i)

        try :
            cc_mean = np.mean(cc, axis=0)
            cc_std = np.std(cc, axis=0)
            ax.errorbar(x = x, y=cc_mean, yerr=cc_std,
                        capthick=1.7, capsize=5,elinewidth=1.7, color="black",
                        errorevery=np.max([len(cc_mean) //10,1]), label="Average")
        except TypeError:
            pass
        plotter.legend()
        plotter.show()

    def _plotRMSDts(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "Time (ps)", "RMSD ($\AA$)")

        # Get matching atoms
        if self.referencePDB.get() != "":
            ref_pdb = PDBMol(self.referencePDB.get())
        else:
            ref_pdb = PDBMol(self.protocol.getInputPDBprefix()+".pdb")
        target_pdb = PDBMol(self.getTargetPDB(1))
        idx = matchPDBatoms([ref_pdb, target_pdb], ca_only=True)

        # Get RMSD list
        fitlist = self.getFitlist()
        rmsd = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i-1)
            for j in outputPrefix:
                rmsd.append(rmsdFromDCD(outputPrefix=j, inputPDB=self.protocol.getInputPDBprefix(i-1)+".pdb",
                    targetPDB=self.getTargetPDB(i),idx=idx, align = self.alignTarget.get()))

        # Plot RMSD
        for i in range(len(rmsd)):
            x = np.arange(len(rmsd[i])) * \
                (int(self.protocol.crdout_period.get())) * \
                float(self.protocol.time_step.get())
            if len(rmsd) <=50:
                ax.plot(x, rmsd[i], alpha=0.3, label="#%i"%i)

        try :
            rmsd_mean = np.mean(rmsd, axis=0)
            rmsd_std = np.std(rmsd, axis=0)
            ax.errorbar(x = x, y=rmsd_mean, yerr=rmsd_std,
                        capthick=1.7, capsize=5,elinewidth=1.7, color="black",
                        errorevery=np.max([len(rmsd_mean) //10,1]), label="Average")
        except TypeError:
            pass
        plotter.legend()
        plotter.show()

    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "# Simulation", "RMSD ($\AA$)")

        # Get RMSD list
        fitlist = self.getFitlist()
        initial_mols = []
        final_mols = []
        target_mols = []
        for i in fitlist:
            inputPDB = self.protocol.getInputPDBprefix(i-1)+".pdb"
            targetPDB = self.getTargetPDB(i)
            outputPrefs = self.protocol.getOutputPrefixAll(i-1)
            target_mols.append(PDBMol(targetPDB))
            initial_mols.append(PDBMol(inputPDB))
            for outputPrefix in outputPrefs:
                outputPDB = outputPrefix +".pdb"
                final_mols.append(PDBMol(outputPDB))

        if self.referencePDB.get() != "":
            ref_mol = PDBMol(self.referencePDB.get())
        else:
            ref_mol = initial_mols[0]
        idx = matchPDBatoms(mols=[ref_mol, target_mols[0]],ca_only=True)
        rmsdi=[]
        rmsdf=[]
        for i in range(len(fitlist)):
            for j in range(len(outputPrefs)):
                rmsdi.append(getRMSD(mol1=initial_mols[i],mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))
                rmsdf.append(getRMSD(mol1=final_mols[i*len(outputPrefs) + j]  ,
                                     mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))

        ax.plot(rmsdf, "o", color="tab:blue", label="Final RMSD", markeredgecolor='black')
        ax.plot(rmsdi, "o", color="tab:green", label="Initial RMSD", markeredgecolor='black')

        plotter.legend()
        plotter.show()

    def _plotPCA(self, paramName):

        initPDB = PDBMol(self.protocol.getInputPDBprefix(0)+".pdb")

        # MAtch atoms with target
        if self.targetPDB.get() is not None:
            targetPDB = PDBMol(self.getTargetPDB(1))
            if self.referencePDB.get() != "":
                refPDB = PDBMol(self.referencePDB.get())
            else:
                refPDB = initPDB
            matchingAtoms = matchPDBatoms([refPDB,targetPDB], ca_only=False)
        else:
            matchingAtoms = np.array([np.arange(initPDB.n_atoms)]).T

        # Get Init PDB coords
        initPDBs = []
        for i in range(self.protocol.getNumberOfInputPDB()):
            mol = PDBMol(self.protocol.getInputPDBprefix(i)+".pdb")
            initPDBs.append(mol.coords[matchingAtoms[:,0]].flatten())

        # Get fitted PDBs coords
        fitlist = self.getFitlist()
        fitPDBs = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                mol = PDBMol(j+".pdb")
                fitPDBs.append(mol.coords[matchingAtoms[:,0]].flatten())

        data = fitPDBs + initPDBs
        length=[len(fitPDBs), len(initPDBs)]
        labels=["Fitted PDBs", "Init. PDBs"]

        # Get TargetPDBs coords
        if self.targetPDB.get() is not None:
            targetPDBs=[]
            for i in fitlist:
                targetPDBs.append(PDBMol(self.getTargetPDB(i)).coords[matchingAtoms[:,1]].flatten())
            data = data+targetPDBs
            length.append(len(targetPDBs))
            labels.append("Target PDBs")

        # Compute PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(np.array(data)).T

        # Plot PCA data
        idx_cumsum = np.concatenate((np.array([0]), np.cumsum(length))).astype(int)
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("PCA", "PCA component 1", "PCA component 2")
        for i in range(len(length)):
            plotter.plot(pca_components[0, idx_cumsum[i]:idx_cumsum[i + 1]],
                         pca_components[1, idx_cumsum[i]:idx_cumsum[i + 1]],
                         "o", label=labels[i],
                         markeredgecolor='black')
        plotter.legend()
        plotter.show()
        fig = plotter.getFigure()

        # Prepare onclick event
        click_coord = []
        inv_pca = []
        n_inv_pca = 10
        initPDB.select_atoms(matchingAtoms[:,0])

        def onclick(event):
            if len(click_coord) < 2:
                click_coord.append((event.xdata, event.ydata))
                x = event.xdata
                y = event.ydata

            if len(click_coord) == 2:
                click_sel = np.array([np.linspace(click_coord[0][0], click_coord[1][0], n_inv_pca),
                                      np.linspace(click_coord[0][1], click_coord[1][1], n_inv_pca)
                                      ])
                ax.plot(click_sel[0], click_sel[1], "-o", color="black")
                inv_pca.insert(0, pca.inverse_transform(click_sel.T))
                click_coord.clear()
                fig.canvas.draw()

                initdcdcp = initPDB.copy()
                coords_list = []
                for i in range(n_inv_pca):
                    coords_list.append(inv_pca[0][i].reshape((initdcdcp.n_atoms, 3)))
                tmpPath = self.protocol._getTmpPath("traj")
                save_dcd(mol=initdcdcp, coords_list=coords_list, prefix=tmpPath)
                initdcdcp.coords = coords_list[0]
                initdcdcp.save(tmpPath+".pdb")
                vmdviewer = VmdView("%s.pdb %s.dcd"%(tmpPath, tmpPath))
                vmdviewer.show()

        fig.canvas.mpl_connect('button_press_event', onclick)

        np.save(file = self.protocol._getExtraPath("PCA_data.npy"), arr= data)
        np.save(file = self.protocol._getExtraPath("PCA_length.npy"), arr= length)
        np.save(file = self.protocol._getExtraPath("PCA_labels.npy"), arr= labels)


    def getFitlist(self):
        return np.array(getListFromRangeString(self.fitRange.get()))

    def getTargetPDB(self, index):
        targetPDBlist = [f for f in glob.glob(self.targetPDB.get())]
        targetPDBlist.sort()
        if index-1 < len(targetPDBlist):
            return targetPDBlist[index-1]
        else:
            return targetPDBlist[0]

