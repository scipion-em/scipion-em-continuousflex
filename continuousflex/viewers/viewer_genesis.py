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
from continuousflex.protocols.protocol_genesis import *
from continuousflex.protocols.utilities.genesis_utilities import *
from .plotter import FlexPlotter
from pwem.viewers import VmdView, ChimeraView
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
from matplotlib.pyplot import cm

COMPARE_PDB_INIT = 0
COMPARE_PDB_OTHER = 1

class FlexGenesisViewer(ProtocolViewer):
    """ Visualization of results from the GENESIS protocol
    """
    _label = 'MD-NMMD-Genesis Viewer'
    _targets = [FlexProtGenesis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        if self.protocol.getNumberOfSimulation() >1:
            form.addParam('fitRange', params.NumericRangeParam,
                          label="Simulation selection",
                          default="1-%i"%self.protocol.getNumberOfSimulation(),
                          important = True,
                          help=' Select the simulation to display. Examples:'
                               ' "1,3-5" -> [1,3,4,5]'
                               ' "1, 2, 4" -> [1,2,4]')
        if self.protocol.simulationType.get() == SIMULATION_REMD\
                or self.protocol.simulationType.get() == SIMULATION_RENMMD:
            form.addParam('replicaRange', params.NumericRangeParam,
                          label="Replica selection",
                          default="1-%i"%self.protocol.nreplica.get(),
                          help=' Select the replicas to display. Examples:'
                               ' "1,3-5" -> [1,3,4,5]'
                               ' "1, 2, 4" -> [1,2,4]')

        form.addParam("numberCurvePerPlot",params.IntParam,default = 10,
                      label="Number of curve per plot ", help="Defines the maximum number of curve"
                      " per plot before averaging into one curve",expertLevel=params.LEVEL_ADVANCED
                      )
        group = form.addGroup('Chimera 3D view')
        group.addParam('displayChimera', params.LabelParam,
                      label='Display results in Chimera',
                      help='Show initial and final structures in Chimera')

        group = form.addGroup('VMD trajectory view')
        group.addParam('displayTrajVMD', params.LabelParam,
                      label='Display trajecory in VMD',
                      help='TODO')

        group = form.addGroup('Energy Analysis')
        group.addParam('displayEnergy', params.LabelParam,
                      label='Display Potential Energy',
                      help='Show time series of the potentials used in MD simulation/Minimization')

        group = form.addGroup('RMSD analysis')
        group.addParam('compareToPDB', params.EnumParam, default=COMPARE_PDB_INIT,
                      label="Compare to ", choices=['initial PDB', 'another PDB'],
                      help='Perform RMSD between the trajectory and another PDB')
        group.addParam('targetPDB', params.PathParam, default=None,
                        label="Target PDB (s)", important=True,
                        help=' Target PDBs to compute RMSD against. Atom mathcing is performed between '
                             ' the output PDBs and the target PDBs. Use the file pattern as file location with /*.pdb',
                       condition= "compareToPDB==%i"%COMPARE_PDB_OTHER)
        group.addParam('referencePDB', params.PathParam, default="",
                        label="Intial PDB (optional)",
                        help='Atom matching will replace the structural information of the output PDBs by the new PDB ',
                        expertLevel=params.LEVEL_ADVANCED,condition= "compareToPDB==%i"%COMPARE_PDB_OTHER)

        group.addParam('alignTarget', params.BooleanParam, default=False,
                        label="Align Target PDB",
                        help='Rigid body align (rotation +translation) the PDBs before RMSD analysis',
                       condition= "compareToPDB==%i"%COMPARE_PDB_OTHER)

        group.addParam('displayRMSDts', params.LabelParam,
                      label='Display RMSD time series')

        group.addParam('displayRMSD', params.LabelParam,
                      label='Display final RMSD')

        if self.protocol.simulationType.get() == SIMULATION_NMMD\
                or self.protocol.simulationType.get() == SIMULATION_RENMMD:
            group = form.addGroup('NMMD')
            group.addParam('displayNMMD', params.LabelParam,
                          label='Display normal modes time series',
                          help='Show normal mode amplitude time series during the simulation')

        if self.protocol.EMfitChoice.get() != EMFIT_NONE:
            group = form.addGroup('Cryo EM fitting')
            group.addParam('displayCC', params.LabelParam,
                          label='Display Correlation Coefficient',
                          help='Show C.C. time series during the simulation')

    def _getVisualizeDict(self):
        return {
            'displayChimera': self._plotChimera,
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayNMMD': self._plotNMMD,
            'displayRMSDts': self._plotRMSDts,
            'displayRMSD': self._plotRMSD,
            'displayTrajVMD': self._plotTrajVMD,
                }

    def _plotChimera(self, paramName):
        tmpChimeraFile = self.protocol._getExtraPath("chimera.cxc")
        index = self.getSimulationList()[0]

        with open(tmpChimeraFile, "w") as f:
            f.write("open %s.pdb \n"% os.path.abspath(self.protocol.getInputPDBprefix(index)))
            f.write("color #1 magenta \n" )
            count = 1

            outpdbfile = self.getOutputPrefixAll(index)[0] +".pdb"
            if os.path.exists(outpdbfile):
                if os.path.getsize(outpdbfile) != 0:
                    f.write("open %s \n"% os.path.abspath(outpdbfile))
                    count+=1
                    f.write("color #%s lime \n"%count)

            if self.compareToPDB.get() == COMPARE_PDB_OTHER:
                f.write("open %s \n" % os.path.abspath(self.getTargetPDB(index)))
                count+=1
                f.write("color #%s orange \n"%count)

            if self.protocol.EMfitChoice.get() == EMFIT_VOLUMES:
                f.write("open %s.mrc \n" % os.path.abspath(self.protocol.getInputEMprefix(index)))
                count+=1
                f.write("volume #%i transparency 0.5\n"%count)

            f.write("hide atoms \n")
            f.write("show cartoons \n")
            f.write("lighting soft \n")

        cv = ChimeraView(tmpChimeraFile)
        cv.show()

    def _plotTrajVMD(self, paramName):
        tmpVmdFile = self.protocol._getExtraPath("vmd.tcl")
        index = self.getSimulationList()[0]
        with open(tmpVmdFile, "w") as f:
            f.write("mol new %s.pdb waitfor all\n" % self.protocol.getInputPDBprefix(index))
            f.write("mol addfile %s.dcd waitfor all\n" % self.getOutputPrefixAll(index)[0])

            if self.protocol.getForceField()== FORCEFIELD_CHARMM:
                f.write("mol modstyle 0 0 NewCartoon \n")
            else:
                f.write("mol modstyle 0 0 Tube \n")
            f.write("mol modcolor 0 0 Molecule\n")

            if self.protocol.EMfitChoice.get() == EMFIT_VOLUMES:
                f.write("mol addfile %s.mrc waitfor all\n" % self.protocol.getInputEMprefix(index))
                f.write("mol addrep 0 \n")
                f.write("mol modstyle 1 0 Isosurface 0.5 0 0 0 1 1  \n")
                f.write("mol modmaterial 1 0 Transparent \n")

            if self.compareToPDB.get() == COMPARE_PDB_OTHER:
                targetFile = self.getTargetPDB(index)
                f.write("set nf [molinfo top get numframes]\n")
                f.write("mol new %s waitfor all\n" %targetFile)
                f.write( "for {set i 1 } {$i < $nf} {incr i} {\n")
                f.write( "animate dup frame 0 1\n")
                f.write("}\n")
                if self.protocol.getForceField() == FORCEFIELD_CHARMM:
                    f.write("mol modstyle 0 1 NewCartoon \n")
                else:
                    f.write("mol modstyle 0 1 Tube \n")
                f.write("mol modcolor 0 1 Molecule\n")
            f.write("animate style Loop\n")
            f.write("display projection Orthographic\n")

        vmdviewer = VmdView(" -e " + tmpVmdFile)
        vmdviewer.show()

    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        ene = {}
        for i in self.getSimulationList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]
        enelist =[]
        labels=[]
        for e in ene :
            labels.append(e)
            enelist.append(ene[e])


        self.genesisPlotter(title="Energy (kcal/mol)", data=enelist, ndata=len(enelist),
                            nrep=len(enelist[0]), labels=labels)
    def _plotNMMD(self, paramName):
        nm = {}
        for i in self.getSimulationList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                for e in range(self.protocol.getNumberOfNormalModes()):
                    nm_name = "NM_AMP%s"%str(e+1).zfill(3)
                    if nm_name in log_file:
                        if nm_name in nm :
                            nm[nm_name].append(log_file[nm_name])
                        else:
                            nm[nm_name] = [log_file[nm_name]]
        nmlist =[]
        labels=[]
        for i in nm :
            labels.append(i)
            nmlist.append(nm[i])

        self.genesisPlotter(title="Normal Mode Amplitude", data=nmlist, ndata=len(nmlist),
                            nrep=len(nmlist[0]), labels=labels)
    def _plotEnergyDetail(self):
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT", "RESTRAINT_TOTAL"]

        ene = {}
        for i in self.getSimulationList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                log_file = readLogFile(j+".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]
        enelist =[]
        labels=[]
        for e in ene :
            labels.append(e)
            enelist.append(ene[e])

        self.genesisPlotter(title="Energy (kcal/mol)", data=enelist, ndata=len(enelist),
                            nrep=len(enelist[0]), labels=labels)

    def _plotCC(self, paramName):
        # Get CC list
        cc = []
        labels=[]
        simlist = self.getSimulationList()
        for i in simlist:
            outputPrefix = self.getOutputPrefixAll(i)
            cc_rep = []
            if len(simlist) == 1:
                labels.append("CC")
            else:
                labels.append("CC %s" % str(i + 1))
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                if 'RESTR_CVS001' in log_file:
                    cc_rep.append(log_file['RESTR_CVS001'])
                else:
                    raise RuntimeError("CC not present in the log file")
            cc.append(cc_rep)

        self.genesisPlotter(title="CC", data=cc, ndata=len(simlist),
                            nrep=len(self.getOutputPrefixAll()), labels=labels)


    def genesisPlotter(self, title, data, ndata, nrep, labels):
        plotter = FlexPlotter()
        nmax = self.numberCurvePerPlot.get()
        ax = plotter.createSubPlot(title, "", title)
        colors = [cm.get_cmap("tab10", nmax)(i) for i in range(nmax) ]

        for i in range(ndata):
            if ndata <= nmax and nrep > 1:
                try:
                    meandata = np.mean(data[i], axis=0)
                    stddata = np.std(data[i], axis=0)
                    x = self.getTimePeriod(len(meandata))
                    ax.errorbar(x=x, y=meandata, yerr=stddata,
                                capthick=1.7, capsize=5, elinewidth=1.7, color=colors[i] if ndata!= 1 else "black",
                                errorevery=np.max([len(meandata) // 10, 1]), label="%s Average" % labels[i])
                except TypeError:
                    x = self.getTimePeriod(len(data[i][0]))
                    ax.plot(x, data[i][0], color=colors[i], label=labels[i])

            for j in range(nrep):
                x = self.getTimePeriod(len(data[i][j]))
                if 1 < nrep <= nmax:
                    if ndata == 1 :
                        ax.plot(x, data[i][j], color= colors[j], alpha=0.5, label="#%i"%(j+1))
                if nrep == 1 and ndata <= nmax:
                    ax.plot(x, data[i][j], color= colors[i],label=labels[i])
        if ndata > nmax :
            try:
                meandata = np.mean(data, axis=(0,1))
                stddata = np.std(data, axis=(0,1))
                x = self.getTimePeriod(len(meandata))
                ax.errorbar(x=x, y=meandata, yerr=stddata,
                            capthick=1.7, capsize=5, elinewidth=1.7, color="black",
                            errorevery=np.max([len(meandata) // 10, 1]), label="Global Average")
            except TypeError:
                x = self.getTimePeriod(len(data[0][0]))
                ax.plot(x, data[0][0], color=colors[0], label=labels[0])
        xlabel = "Number of iterations" if self.protocol.simulationType.get() == SIMULATION_MIN else "Time (ps)"
        ax.set_xlabel(xlabel)
        plotter.legend()
        plotter.show()

    def _plotRMSDts(self, paramName):

        # Get matching atoms
        if self.referencePDB.get() != "":
            ref_pdb = ContinuousFlexPDBHandler(self.referencePDB.get())
        else:
            ref_pdb = ContinuousFlexPDBHandler(self.protocol.getInputPDBprefix()+".pdb")
        if self.compareToPDB.get()  == 1:
            target_pdb = ContinuousFlexPDBHandler(self.getTargetPDB())
            idx_matchin_atoms = ref_pdb.matchPDBatoms(reference_pdb=target_pdb, ca_only=True)
        else:
            idx_matchin_atoms = None

        # Get RMSD list
        rmsd = []
        labels=[]
        simlist = self.getSimulationList()
        for i in simlist:
            outputPrefix = self.getOutputPrefixAll(i)
            if len(simlist) == 1:
                labels.append("RMSD")
            else:
                labels.append("RMSD %s"%str(i+1))
            rmsd_rep=[]
            for outprf in outputPrefix:
                rmsd_curr = []

                inputPDB = ContinuousFlexPDBHandler(self.protocol.getInputPDBprefix(i)+".pdb")
                if self.compareToPDB.get() == 1:
                    targetPDB = ContinuousFlexPDBHandler(self.getTargetPDB(i))
                else:
                    targetPDB = ref_pdb
                rmsd_curr.append(inputPDB.getRMSD(reference_pdb=targetPDB, align=self.alignTarget.get(), idx_matching_atoms=idx_matchin_atoms))
                coord_arr = dcd2numpyArr(outprf + ".dcd")
                for i in range(len(coord_arr)):
                    inputPDB.coords[:, :] = coord_arr[i]
                    rmsd_curr.append(inputPDB.getRMSD(reference_pdb=targetPDB, align=self.alignTarget.get(), idx_matching_atoms=idx_matchin_atoms))

                rmsd_rep.append(rmsd_curr)
            rmsd.append(rmsd_rep)

        self.genesisPlotter(title="RMSD ($\AA$)", data=rmsd, ndata=len(simlist),
                            nrep=len(self.getOutputPrefixAll()), labels=labels)


    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "# Simulation", "RMSD ($\AA$)")

        # Get RMSD list
        initial_mols = []
        final_mols = []
        target_mols = []
        for i in self.getSimulationList():
            inputPDB = self.protocol.getInputPDBprefix(i)+".pdb"
            if self.compareToPDB.get() == 1:
                targetPDB = self.getTargetPDB(i)
            else:
                targetPDB = inputPDB
            outputPrefs = self.getOutputPrefixAll(i)
            target_mols.append(ContinuousFlexPDBHandler(targetPDB))
            initial_mols.append(ContinuousFlexPDBHandler(inputPDB))
            for outputPrefix in outputPrefs:
                outputPDB = outputPrefix +".pdb"
                final_mols.append(ContinuousFlexPDBHandler(outputPDB))

        if self.referencePDB.get() != "":
            ref_mol = ContinuousFlexPDBHandler(self.referencePDB.get())
        else:
            ref_mol = initial_mols[0]
        if self.compareToPDB.get() == 1:
            idx_match = ref_mol.matchPDBatoms(reference_pdb=target_mols[0],ca_only=True)
        else:
            idx_match = None
        rmsdi=[]
        rmsdf=[]
        for i in range(len(self.getSimulationList())):
            for j in range(len(outputPrefs)):
                rmsdi.append(initial_mols[i].getRMSD(reference_pdb=target_mols[i],
                                                     idx_matching_atoms=idx_match,
                                                     align=self.alignTarget.get()))
                rmsdf.append(final_mols[i*len(outputPrefs) + j].getRMSD(reference_pdb=target_mols[i],
                                                    idx_matching_atoms=idx_match, align=self.alignTarget.get()))

        ax.plot(rmsdf, "o", color="tab:blue", label="Final RMSD", markeredgecolor='black')
        ax.plot(rmsdi, "o", color="tab:green", label="Initial RMSD", markeredgecolor='black')
        plotter.legend()
        plotter.show()

    def getSimulationList(self):
        if self.protocol.getNumberOfSimulation() > 1:
            return np.array(getListFromRangeString(self.fitRange.get())) -1
        else:
            return np.array([0])

    def getTargetPDB(self, index=0):
        targetPDBlist = [f for f in glob.glob(self.targetPDB.get())]
        targetPDBlist.sort()
        if index < len(targetPDBlist):
            return targetPDBlist[index]
        else:
            return targetPDBlist[0]

    def getOutputPrefixAll(self, index=0):
        outPrf = np.array(self.protocol.getOutputPrefixAll(index))
        if self.protocol.simulationType.get() == SIMULATION_REMD \
                or self.protocol.simulationType.get() == SIMULATION_RENMMD:
            return outPrf[np.array(getListFromRangeString(self.replicaRange.get())) - 1]
        else:
            return outPrf

    def getTimePeriod(self, length):
        if self.protocol.simulationType.get() == SIMULATION_MIN:
            timestep = 1.0
        else:
            timestep = float(self.protocol.time_step.get())
        eneperiod = int(self.protocol.eneout_period.get())

        return np.arange(length) * eneperiod * timestep

