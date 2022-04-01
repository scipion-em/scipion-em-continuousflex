# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
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


from os.path import basename
import numpy as np
from pwem.emlib import MetaData, MDL_ORDER
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceBaseExt, replaceExt

from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtDimredPdb
import xmipp3
import pwem.emlib.metadata as md
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt

from joblib import load
from continuousflex.viewers.nma_vol_gui import TrajectoriesWindowVol
from continuousflex.protocols.data import Point, Data, PathData
from pwem.viewers import VmdView
from pyworkflow.utils.path import cleanPath, makePath
from continuousflex.protocols.utilities.genesis_utilities import PDBMol,save_dcd
from pyworkflow.gui.browser import FileBrowserWindow

import os

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1

NUM_POINTS_TRAJECTORY=10


class FlexProtPdbDimredViewer(ProtocolViewer):
    """ Visualization of dimensionality reduction on PDBs
    """
    _label = 'viewer PDBs dimred'
    _targets = [FlexProtDimredPdb]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayRawDeformation', StringParam, default='1 2',
                      label='Display the principal axes',
                      help='Type 1 to see the histogram of PCA axis 1; \n'
                           'type 2 to to see the histogram of PCA axis 2, etc.\n'
                           'Type 1 2 to see the 2D plot of amplitudes for PCA axes 1 2.\n'
                           'Type 1 2 3 to see the 3D plot of amplitudes for PCA axes 1 2 3; etc.'
                           )
        form.addParam('displayPcaSingularValues', LabelParam,
                      label="Display PCA singular values",
                      help="The values should help you see how many dimensions are in the data ")
        form.addParam('displayTrajectories', LabelParam,
                      label='Open trajectories tool?',
                      help='Open a GUI to visualize the volumes as points'
                           ' to draw and adjust trajectories.')
        form.addParam('xlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually x-axis limits'],
                      default=X_LIMITS_NONE,
                      label='x-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of x-axis limits')
        form.addParam('xlim_low', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Lower x-axis limit')
        form.addParam('xlim_high', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Upper x-axis limit')
        form.addParam('ylimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually y-axis limits'],
                      default=Y_LIMITS_NONE,
                      label='y-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of y-axis limits')
        form.addParam('ylim_low', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Lower y-axis limit')
        form.addParam('ylim_high', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Upper y-axis limit')
        form.addParam('zlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually z-axis limits'],
                      default=Z_LIMITS_NONE,
                      label='z-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of z-axis limits')
        form.addParam('zlim_low', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Lower z-axis limit')
        form.addParam('zlim_high', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Upper z-axis limit')
        form.addParam('s', FloatParam, default=None, allowsNull=True,
                      label='Radius')
        form.addParam('alpha', FloatParam, default=None, allowsNull=True,
                        label='Transparancy')


    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayPcaSingularValues': self.viewPcaSinglularValues,
                'displayTrajectories': self._displayTrajectories,
                }


    def _displayTrajectories(self, paramName):
        self.trajectoriesWindow = self.tkWindow(TrajectoriesWindowVol,
                                                title='Trajectories Tool',
                                                dim=self.protocol.reducedDim.get(),
                                                data=self.getData(),
                                                callback=self._generateAnimation,
                                                loadCallback=self._loadAnimation,
                                                numberOfPoints=NUM_POINTS_TRAJECTORY,
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
                                                alpha=self.alpha)
        return [self.trajectoriesWindow]

    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)
        
    def _doViewRawDeformation(self, components):
        components = list(map(int, components.split()))
        # print(components)
        dim = len(components)
        if self.xlimits_mode.get() == X_LIMITS:
            x_low = self.xlim_low.get()
            x_high = self.xlim_high.get()
        if self.ylimits_mode.get() == Y_LIMITS:
            y_low = self.ylim_low.get()
            y_high = self.ylim_high.get()
        if self.zlimits_mode.get() == Z_LIMITS:
            z_low = self.zlim_low.get()
            z_high = self.zlim_high.get()

        # print(self.protocol.getOutputMatrixFile())
        X = np.loadtxt(fname=self.protocol.getOutputMatrixFile())
        if dim == 1:
            plt.hist(X[:,components[0]-1])
            plt.title('Histogram of principal axis %d values' %components[0])
        if dim == 2:
            plt.scatter(X[:,components[0]-1],X[:,components[1]-1])
            if self.xlimits_mode.get() == X_LIMITS:
                plt.xlim([x_low,x_high])
            if self.ylimits_mode.get() == Y_LIMITS:
                plt.ylim([y_low,y_high])
            plt.xlabel('Principal Component Axis %d' %components[0])
            plt.ylabel('Principal Component Axis %d' %components[1])
            modeNameList = 'Principal Axes %d vs %d' %(components[0], components[1])
            plt.title(modeNameList)

        if dim == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(X[:,components[0]-1],X[:,components[1]-1],X[:,components[2]-1])
            if self.xlimits_mode.get() == X_LIMITS:
                ax.set_xlim([x_low,x_high])
            if self.ylimits_mode.get() == Y_LIMITS:
                ax.set_ylim([y_low,y_high])
            if self.zlimits_mode.get() == Z_LIMITS:
                ax.set_zlim([z_low,z_high])
            ax.set_xlabel('Principal Component Axis %d' %components[0])
            ax.set_ylabel('Principal Component Axis %d' %components[1])
            ax.set_zlabel('Principal Component Axis %d' %components[2])
            modeNameList = 'Principal Axes %d vs %d vs %d' %(components[0], components[1], components[2])
            ax.set_title(modeNameList)
        plt.show()

    def viewPcaSinglularValues(self, paramName):
        pca = load(self.protocol._getExtraPath('pca_pickled.joblib'))
        fig = plt.figure('PCA singlular values')
        plt.stem(pca.singular_values_)
        plt.xticks(np.arange(0, len(pca.singular_values_), 1))
        plt.show()
        pass

    def getData(self):
        data = Data()
        pdb_matrix = np.loadtxt(self.protocol.getOutputMatrixFile())
        for i in range(pdb_matrix.shape[0]):
            data.addPoint(Point(pointId=i+1, data=pdb_matrix[i, :],weight=1.0))
        return data

    def _generateAnimation(self):
        prot = self.protocol

        # Get animation root
        animation = self.trajectoriesWindow.getAnimationName()
        animationPath = prot._getExtraPath('animation_%s' % animation)
        cleanPath(animationPath)
        makePath(animationPath)
        animationRoot = os.path.join(animationPath, 'animation_%s' % animation)

        # get trajectory coordinates
        trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])
        np.savetxt(animationRoot + 'trajectory.txt', trajectoryPoints)
        if prot.getMethodName() == 'sklearn_PCA':
            pca = load(prot._getExtraPath('pca_pickled.joblib'))
            deformations = pca.inverse_transform(trajectoryPoints)
        else:
            projectorFile = prot._getExtraPath() + '/projector.txt'
            if os.path.isfile(projectorFile):
                M = np.loadtxt(projectorFile)
                deformations = np.dot(trajectoryPoints, np.linalg.pinv(M))
                temp = np.loadtxt(prot._getExtraPath('deformations.txt'))  # the original matrix file
                deformations += np.outer(np.ones(deformations.shape[0]), np.mean(temp, axis=0))

            else:
                Y = np.loadtxt(prot.getOutputMatrixFile())
                X = np.loadtxt(prot.getDeformationFile())
                # Find closest points in deformations
                deformations = [X[np.argmin(np.sum((Y - p) ** 2, axis=1))] for p in trajectoryPoints]

        # Generate DCD trajectory
        initPDB = PDBMol(prot.getPDBRef())
        initdcdcp = initPDB.copy()
        coords_list = []
        for i in range(NUM_POINTS_TRAJECTORY):
            coords_list.append(deformations[i].reshape((initdcdcp.n_atoms, 3)))
        save_dcd(mol=initdcdcp, coords_list=coords_list, prefix=animationRoot)
        initdcdcp.coords = coords_list[0]
        initdcdcp.save(animationRoot+".pdb")

        # Generate the vmd script
        vmdFn = animationRoot + '.vmd'
        vmdFile = open(vmdFn, 'w')
        vmdFile.write("""
        mol load pdb %s.pdb dcd %s.dcd
        animate style Rock
        display projection Orthographic
        mol modcolor 0 0 Index
        mol modstyle 0 0 Tube 1.000000 8.000000
        animate speed 1.0
        animate forward
        """ % (animationRoot,animationRoot))
        vmdFile.close()

        VmdView(' -e ' + vmdFn).show()

    def _loadAnimation(self):
        browser = FileBrowserWindow("Select the animation folder (animation_NAME)",
                                    self.getWindow(), self.protocol._getExtraPath(),
                                    onSelect=self._loadAnimationData)
        browser.show()

    def _loadAnimationData(self, obj):
        prot = self.protocol
        animationName = obj.getFileName()  # assumes that obj.getFileName is the folder of animation
        animationPath = prot._getExtraPath(animationName)
        animationRoot = os.path.join(animationPath, animationName)

        animationSuffixes = ['.vmd', '.pdb','.dcd', 'trajectory.txt']
        for s in animationSuffixes:
            f = animationRoot + s
            if not os.path.exists(f):
                self.errorMessage('Animation file "%s" not found. ' % f)
                return

        # Load animation trajectory points
        trajectoryPoints = np.loadtxt(animationRoot + 'trajectory.txt')
        data = PathData(dim=trajectoryPoints.shape[1])

        for i, row in enumerate(trajectoryPoints):
            data.addPoint(Point(pointId=i + 1, data=list(row), weight=1))

        self.trajectoriesWindow.setPathData(data)
        self.trajectoriesWindow.setAnimationName(animationName)
        self.trajectoriesWindow._onUpdateClick()

        def _showVmd():
            vmdFn = animationRoot + '.vmd'
            VmdView(' -e %s' % vmdFn).show()

        self.getTkRoot().after(500, _showVmd)

