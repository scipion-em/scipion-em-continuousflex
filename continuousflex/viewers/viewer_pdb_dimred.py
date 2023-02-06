# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Remi Vuillemot             (remi.vuillemot@upmc.fr)
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
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam, PointerParam, IntParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pwem.viewers import ChimeraView
from pwem.objects.data import SetOfParticles,SetOfVolumes
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtDimredPdb
import matplotlib.pyplot as plt
from pwem.emlib.image import ImageHandler
from joblib import load
from continuousflex.viewers.tk_dimred import PCAWindowDimred
from continuousflex.protocols.data import Point, Data, PathData
from pwem.viewers import VmdView
from pyworkflow.utils.path import cleanPath, makePath
from continuousflex.protocols.utilities.genesis_utilities import numpyArr2dcd, dcd2numpyArr
from continuousflex.protocols.utilities.pdb_handler import ContinuousFlexPDBHandler
from pyworkflow.gui.browser import FileBrowserWindow
from continuousflex.protocols.protocol_pdb_dimred import REDUCE_METHOD_PCA, REDUCE_METHOD_UMAP
from continuousflex.protocols.protocol_batch_pdb_cluster import FlexBatchProtClusterSet
from .plotter import FlexPlotter
import os

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1

ANIMATION_INV=0
ANIMATION_AVG=1
ANIMATION_PCA=2

NUM_POINTS_TRAJECTORY=10


class FlexProtPdbDimredViewer(ProtocolViewer):
    """ Visualization of dimensionality reduction on atomic structures
    """
    _label = 'viewer PDBs dimred'
    _targets = [FlexProtDimredPdb]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        group = form.addGroup("Display Singular Values")
        group.addParam('displayPcaSingularValues', LabelParam,
                      label="Display singular values",
                      help="The values should help you see how many dimensions are in the data ",
                      condition=self.protocol.method.get()==REDUCE_METHOD_PCA)

        group = form.addGroup("Display PCA")
        group.addParam('displayPCA', LabelParam,
                      label='Display PCA axes',
                      help='Open a GUI to visualize the PCA space')

        group.addParam('pcaAxes', StringParam, default="1 2",
                       label='Axes to display' )

        group = form.addGroup("Display free energy")
        group.addParam('displayFreeEnergy', LabelParam,
                      label='Display free energy',
                      help='Open a GUI to visualize the PCA space as free energy landscape')
        group.addParam('freeEnergyAxes', StringParam, default="1 2",
                       label='Axes to display' )
        group.addParam('freeEnergySize', IntParam, default=100,
                       label='Sampling size' )

        group = form.addGroup("Animation tool")

        group.addParam('displayAnimationtool', LabelParam,
                      label='Open Animation tool ',
                      help='Open a GUI to analyze the PCA space'
                           ' to draw and adjust trajectories and create clusters.')

        group.addParam('inputSet', PointerParam, pointerClass ='SetOfParticles,SetOfVolumes',
                      label='(Optional) Em data for cluster animation',  allowsNull=True,
                      help="Provide a EM data set that match the PDB data set to visualize animation on 3D reconstructions")

        group = form.addGroup("Figure parameters")

        group.addParam('s', FloatParam, default=10, allowsNull=True,
                       label='Point radius')
        group.addParam('alpha', FloatParam, default=0.5, allowsNull=True,
                       label='Point transparancy')
        group.addParam('xlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually x-axis limits'],
                      default=X_LIMITS_NONE,
                      label='x-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of x-axis limits')
        group.addParam('xlim_low', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Lower x-axis limit')
        group.addParam('xlim_high', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Upper x-axis limit')
        group.addParam('ylimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually y-axis limits'],
                      default=Y_LIMITS_NONE,
                      label='y-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of y-axis limits')
        group.addParam('ylim_low', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Lower y-axis limit')
        group.addParam('ylim_high', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Upper y-axis limit')
        group.addParam('zlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually z-axis limits'],
                      default=Z_LIMITS_NONE,
                      label='z-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of z-axis limits')
        group.addParam('zlim_low', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Lower z-axis limit')
        group.addParam('zlim_high', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Upper z-axis limit')


    def _getVisualizeDict(self):
        return {
                'displayPCA': self._displayPCA,
                'displayFreeEnergy': self._displayFreeEnergy,
                'displayAnimationtool': self._displayAnimationtool,
                'displayPcaSingularValues': self.viewPcaSinglularValues,
                }

    def _displayPCA(self, paramName):
        axes_str = str.split(self.pcaAxes.get())
        axes = []
        for i in axes_str : axes.append(int(i.strip()))

        dim = len(axes)
        if dim ==0 or dim >3:
            return self.errorMessage("Can not read input PCA axes selection", "Invalid Input")

        data = self.getData()
        plotter = FlexNmaPlotter(data= data,
                                      xlim_low=self.xlim_low.get(), xlim_high=self.xlim_high.get(),
                                      ylim_low=self.ylim_low.get(), ylim_high=self.ylim_high.get(),
                                      zlim_low=self.zlim_low.get(), zlim_high=self.zlim_high.get(),
                                      alpha=self.alpha, s=self.s, cbar_label=None)
        if dim == 1:
            data.XIND = axes[0]-1
            plotter.plotArray1D("PCA","%i component"%(axes[0]),"")
        if dim == 2:
            data.YIND = axes[1]-1
            plotter.plotArray2D_xy("PCA","%i component"%(axes[0]),"%i component"%(axes[1]))
        if dim == 3:
            data.ZIND = axes[2]-1
            plotter.plotArray3D_xyz("PCA","%i component"%(axes[0]),"%i component"%(axes[1]),"%i component"%(axes[2]))
        plotter.show()

    def _displayFreeEnergy(self, paramName):
        axes_str = str.split(self.freeEnergyAxes.get())
        axes = []
        for i in axes_str : axes.append(int(i.strip())-1)

        dim = len(axes)
        if dim != 2:
            return self.errorMessage("Please select only 2 axes", "Invalid Input")

        data = np.array([p.getData()[axes] for p in self.getData()])
        size =self.freeEnergySize.get()
        xmin = np.min(data[:,0])
        xmax = np.max(data[:,0])
        ymin = np.min(data[:,1])
        ymax = np.max(data[:,1])
        x = np.linspace(xmin, xmax, size)
        y = np.linspace(ymin, ymax, size)
        count = np.zeros((size, size))
        for i in range(data.shape[0]):
            count[np.argmin(np.abs(x.T - data[i, 0])),
                  np.argmin(np.abs(y.T - data[i, 1]))] += 1
        img = -np.log(count / count.max())
        img[img == np.inf] = img[img != np.inf].max()

        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Free energy", "component "+axes_str[0],
                                        "component " + axes_str[1])
        # im = ax.imshow(img.T[::-1,:],
        #            cmap = "jet", interpolation=interp,
        #            extent=[xmin,xmax,ymin,ymax])

        xx, yy = np.mgrid[xmin:xmax:size * 1j, ymin:ymax:size * 1j]
        im = ax.contourf(xx, yy, img, cmap='jet')
        cbar = plotter.figure.colorbar(im)
        cbar.set_label("$\Delta G / k_{B}T$")
        plotter.show()



    def _displayAnimationtool(self, paramName):
        self.trajectoriesWindow = self.tkWindow(PCAWindowDimred,
                                                title='Animation tool',
                                                dim=self.protocol.reducedDim.get(),
                                                data=self.getData(),
                                                callback=self._generateAnimation,
                                                loadCallback=self._loadAnimation,
                                                saveCallback=self._saveAnimation,
                                                saveClusterCallback=self.saveClusterCallback,
                                                numberOfPoints=5,
                                                limits_mode=0,
                                                LimitL=None,
                                                LimitH=None,
                                                xlim_low=self.xlim_low.get(),
                                                xlim_high=self.xlim_high.get(),
                                                ylim_low=self.ylim_low.get(),
                                                ylim_high=self.ylim_high.get(),
                                                zlim_low=self.zlim_low.get(),
                                                zlim_high=self.zlim_high.get(),
                                                s=self.s,
                                                alpha=self.alpha,
                                                cbar_label="Cluster")
        return [self.trajectoriesWindow]


    def viewPcaSinglularValues(self, paramName):
        pca = load(self.protocol._getExtraPath('pca_pickled.joblib'))
        fig = plt.figure('PCA singlular values')
        plt.stem(np.arange(1, len(pca.singular_values_)+1), pca.singular_values_)
        plt.show()
        pass

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def loadData(self):
        data = Data()
        pdb_matrix = np.loadtxt(self.protocol.getOutputMatrixFile())
        weights = [0 for i in range(pdb_matrix.shape[0])]

        for i in range(pdb_matrix.shape[0]):
            data.addPoint(Point(pointId=i+1, data=pdb_matrix[i, :],weight=weights[i]))
        return data

    def _generateAnimation(self):
        prot = self.protocol
        initPDB = ContinuousFlexPDBHandler(prot.getPDBRef())

        # Get animation root
        animation = self.trajectoriesWindow.getClusterName()
        animationPath = prot._getExtraPath('animation_%s' % animation)
        if not os.path.isdir(animationPath):
            cleanPath(animationPath)
            makePath(animationPath)
        animationRoot = os.path.join(animationPath, '')

        # get trajectory coordinates
        animtype = self.trajectoriesWindow.getAnimationType()
        coords_list = []
        if animtype ==ANIMATION_INV:
            trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])
            if trajectoryPoints.shape[0] == 0 :
                return self.trajectoriesWindow.showError("No animation to show.")
            np.savetxt(animationRoot + 'trajectory.txt', trajectoryPoints)
            pca = load(prot._getExtraPath('pca_pickled.joblib'))
            deformations = pca.inverse_transform(trajectoryPoints)
            for i in range(self.trajectoriesWindow.numberOfPoints):
                coords_list.append(deformations[i].reshape((initPDB.n_atoms, 3)))
        else :
            # read save coordinates
            coords = dcd2numpyArr(self.protocol._getExtraPath("coords.dcd"))

            # get class dict
            classDict = {}
            count = 0 #CLUSTERINGTAG
            for p in self.trajectoriesWindow.data:
                clsId = str(int(p._weight)) #CLUSTERINGTAG
                if clsId in classDict:
                    classDict[clsId].append(count)
                else:
                    classDict[clsId] = [count]
                count += 1

            if animtype == ANIMATION_AVG:
                # compute avg
                for i in classDict:
                    coord_avg = np.mean(coords[np.array(classDict[i])], axis=0)
                    coords_list.append(coord_avg.reshape((initPDB.n_atoms, 3)))


        # Generate DCD trajectory
        initdcdcp = initPDB.copy()
        initdcdcp.coords = coords_list[0]
        initdcdcp.write_pdb(animationRoot+"trajectory.pdb")
        numpyArr2dcd(arr = np.array(coords_list), filename=animationRoot+"trajectory.dcd")

        # Generate the vmd script
        vmdFn = animationRoot + 'trajectory.vmd'
        vmdFile = open(vmdFn, 'w')
        vmdFile.write("""
        mol new %strajectory.pdb waitfor all
        mol addfile %strajectory.dcd waitfor all
        animate style Rock
        display projection Orthographic
        mol modcolor 0 0 Index
        mol modstyle 0 0 Tube 1.000000 8.000000
        animate speed 0.75
        animate forward
        animate delete  beg 0 end 0 skip 0 0
        """ % (animationRoot,animationRoot))
        vmdFile.close()

        VmdView(' -e ' + vmdFn).show()

    def saveClusterCallback(self, tkWindow):
        # get cluster name
        clusterName = "animation_" + tkWindow.getClusterName()

        # get input metadata
        inputSet = self.inputSet.get()
        if inputSet is None:
            tkWindow.showError("Select an EM set before exporting clusters.")
            return

        classID=[]
        for p in tkWindow.data:
            classID.append(int(p._weight))

        if isinstance(inputSet, SetOfParticles):
            classSet = self.protocol._createSetOfClasses2D(inputSet, clusterName)
        else:
            classSet = self.protocol._createSetOfClasses3D(inputSet,clusterName)

        def updateItemCallback(item, row):
            item.setClassId(row)

        class itemDataIterator:
            def __init__(self, clsID):
                self.clsID = clsID

            def __iter__(self):
                self.n = 0
                return self

            def __next__(self):
                if self.n > len(self.clsID)-1:
                    return 0
                else:
                    index = self.clsID[self.n]
                    self.n += 1
                    return index

        classSet.classifyItems(
            updateItemCallback=updateItemCallback,
            updateClassCallback=None,
            itemDataIterator=iter(itemDataIterator(classID)),
            classifyDisabled=False,
            iterParams=None,
            doClone=True)

        # Run reconstruction
        self.protocol._defineOutputs(**{clusterName : classSet})
        project = self.protocol.getProject()
        newProt = project.newProtocol(FlexBatchProtClusterSet)
        newProt.setObjLabel(clusterName)
        newProt.inputSet.set(getattr(self.protocol, clusterName))
        project.launchProtocol(newProt)
        project.getRunsGraph()

        tkWindow.showInfo("Successfully exported clustering.")

    def _loadAnimation(self):
        browser = FileBrowserWindow("Select animation directory",
                                    self.getWindow(), self.protocol._getExtraPath(),
                                    onSelect=self._loadAnimationData)
        browser.show()

    def _loadAnimationData(self, obj):

        if not obj.isDir() :
            return self.trajectoriesWindow.showError('Not a directory')

        loaded = []
        trajPath = obj.getPath()
        trajFile = os.path.join(trajPath,'trajectory.txt')
        if os.path.isfile(trajFile) and os.path.getsize(trajFile) != 0:
            trajectoryPoints = np.loadtxt(trajFile)
            data = PathData(dim=trajectoryPoints.shape[1])
            for i, row in enumerate(trajectoryPoints):
                data.addPoint(Point(pointId=i + 1, data=list(row), weight=0))
            loaded.append("trajectory.txt")

        clusterFile = os.path.join(trajPath,'clusters.txt')
        if os.path.isfile(clusterFile) and os.path.getsize(clusterFile) != 0:
            clusterPoints = np.loadtxt(clusterFile)
            i=0
            for p in self.trajectoriesWindow.data:
                p._weight = clusterPoints[i]
                i+=1
            loaded.append("clusters.txt")
        if len(loaded) ==0:
            return self.trajectoriesWindow.showError('Animation files not found. ')
        else:
            self.trajectoriesWindow.showInfo('Successfully loaded : %s.' %str(loaded))

        self.trajectoriesWindow.setPathData(data)
        self.trajectoriesWindow._onUpdateClick()
        self.trajectoriesWindow._checkNumberOfPoints()

    def _saveAnimation(self, tkWindow):
        # get cluster name
        animationPath = self.protocol._getExtraPath("animation_" + tkWindow.getClusterName())
        if not os.path.isdir(animationPath):
            cleanPath(animationPath)
            makePath(animationPath)
        animationRoot = os.path.join(animationPath, '')
        trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])
        saved=[]
        if trajectoryPoints.shape[0] != 0:
            np.savetxt(animationRoot + 'trajectory.txt', trajectoryPoints)
            saved.append('trajectory.txt')

        classID=[]
        for p in tkWindow.data:
            classID.append(int(p._weight))
        if set(classID) != {0}:
            np.savetxt(animationRoot + 'clusters.txt', np.array(classID))
            saved.append('clusters.txt')

        try :
            self.trajectoriesWindow.plotter.figure.savefig(animationRoot + 'figure.png',dpi=500)
        except:
            pass

        if len(saved) != 0:
            self.trajectoriesWindow.showInfo('Successfully saved : %s.' % str(saved))
        else:
            self.trajectoriesWindow.showError('No animation state to save.')


class VolumeTrajectoryViewer(ProtocolViewer):
    """ Visualization of a SetOfVolumes as a trajectory with ChimeraX
    """
    _label = 'Volume trajectory viewer'
    _targets = [SetOfVolumes]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayTrajectories', LabelParam,
                      label='ChimeraX',
                      help='Open the trajectory in ChimeraX.')
    def _getVisualizeDict(self):
        return {
                'displayTrajectories': self._visualize,
                }

    def _visualize(self, obj, **kwargs):
        """visualisation for volumes set"""
        volNames = ""
        for i in self.protocol:
            i.setSamplingRate(self.protocol.getSamplingRate())
            vol = ImageHandler().read(i)
            volName = os.path.abspath(self._getPath("tmp%i.vol"%i.getObjId()))
            vol.write(volName)
            volNames += volName+" "
        # Show Chimera
        tmpChimeraFile = self._getPath("chimera.cxc")
        with open(tmpChimeraFile, "w") as f:
            f.write("open %s vseries true \n" % volNames)
            # f.write("volume #1 style surface level 0.5")
            f.write("vseries play #1 loop true maxFrameRate 7 direction oscillate \n")

        cv = ChimeraView(tmpChimeraFile)
        return [cv]
