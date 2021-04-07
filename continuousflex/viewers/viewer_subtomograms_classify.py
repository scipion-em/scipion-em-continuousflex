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
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam, IntParam, LEVEL_ADVANCED
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceBaseExt, replaceExt

from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtSubtomoClassify
import xmipp3
import pwem.emlib.metadata as md
from pyworkflow.utils.process import runJob
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt
from joblib import load
import scipy.cluster.hierarchy as sch

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1


class FlexProtSubtomoClassifyViewer(ProtocolViewer):
    """ Visualization of dimensionality reduction on PDBs
    """
    _label = 'viewer subtomograms classify'
    _targets = [FlexProtSubtomoClassify]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayAgglomarative', LabelParam,
                      label="Display Hierarchical Clustering Tree",
                      help="Display the dendrogram that corresponds to the Hierarchical clustering")
        form.addParam('displayFullAgglomarative', LabelParam,
                      expertLevel=LEVEL_ADVANCED,
                      label="Display Full Hierarchical Clustering Tree",
                      help="Display the full tree without truncating for the first p clusters")
        form.addParam('displayRawDeformation', StringParam, default='1 2',
                      label='Display the principle axes',
                      help='Type 1 to see the histogram of PCA axis 1; \n'
                           'type 2 to to see the histogram of PCA axis 2, etc.\n'
                           'Type 1 2 to see the 2D plot of amplitudes for PCA axes 1 2.\n'
                           'Type 1 2 3 to see the 3D plot of amplitudes for PCA axes 1 2 3; etc.'
                           )
        form.addParam('displayPcaSingularValues', LabelParam,
                      label="Display PCA singular values",
                      help="The values should help you see how many dimensions are in the data ")
        form.addParam('displayKmeans', StringParam, default='1 2',
                      label='Display Kmeans classification on the principle axes',
                      help='Type 1 2 to see the classification 2D plot on the PCA axes 1 2.\n'
                           'Type 1 2 3 to see the classification 3D plot on the PCA axes 1 2 3; etc.'
                           )
        form.addParam('blacked', IntParam,
                      default=None,
                      allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      label='blacked cluster',
                      help='This allows you to make a specific cluster color to black to identify it.'
                           'If 0 this will make cluster 0 black on the graph'
                           'If 1 this will make cluster 1 black on the graph, etc.')
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

    def _getVisualizeDict(self):
        return {'displayAgglomarative': self.viewDendrogram,
                'displayFullAgglomarative': self.viewFullDendrogram,
                'displayRawDeformation': self._viewRawDeformation,
                'displayPcaSingularValues': self.viewPcaSinglularValues,
                'displayKmeans':self._viewKmeans}

    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)

    def _viewKmeans(self, paramName):
        components = self.displayKmeans.get()
        return self._doViewKmeans(components)

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

        X = np.loadtxt(fname=self.protocol._getExtraPath('dimred_mat.txt'))
        if dim == 1:
            plt.hist(X[:,components[0]-1])
        if dim == 2:
            plt.scatter(X[:,components[0]-1],X[:,components[1]-1])
            if self.xlimits_mode.get() == X_LIMITS:
                plt.xlim([x_low,x_high])
            if self.ylimits_mode.get() == Y_LIMITS:
                plt.ylim([y_low,y_high])
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
        plt.show()

    def _doViewKmeans(self,components):
        components = list(map(int, components.split()))
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

        Y = np.loadtxt(fname=self.protocol._getExtraPath('dimred_mat.txt'))
        kmeans = load(self.protocol._getExtraPath('kmeans_algo.pkl'))
        label = kmeans.labels_

        if dim == 2:
            for l in np.unique(label):
                if l == self.blacked.get():
                    color = (0, 0, 0)
                    s = 100
                else:
                    color = plt.cm.jet(float(l) / np.max(label + 1))
                    s = 50
                fig = plt.figure('2D')
                plt.scatter(Y[label == l, components[0]-1], Y[label == l, components[1]-1],
                            color=color,
                            edgecolor='k',
                            s = s)
            if self.xlimits_mode.get() == X_LIMITS:
                plt.xlim([x_low,x_high])
            if self.ylimits_mode.get() == Y_LIMITS:
                plt.ylim([y_low,y_high])
            plt.show()


        if dim == 3:
            for l in np.unique(label):
                if l == self.blacked.get():
                    color = (0, 0, 0)
                    s = 100
                else:
                    color = plt.cm.jet(float(l) / np.max(label + 1))
                    s = 50
                fig = plt.figure('3D')
                ax = fig.gca(projection='3d')
                ax.scatter(Y[label == l, 0], Y[label == l, 1], Y[label == l, 2],
                            color=color,
                            edgecolor='k',
                           s = s)
            if self.xlimits_mode.get() == X_LIMITS:
                ax.set_xlim([x_low,x_high])
            if self.ylimits_mode.get() == Y_LIMITS:
                ax.set_ylim([y_low,y_high])
            if self.zlimits_mode.get() == Z_LIMITS:
                ax.set_zlim([z_low,z_high])
            plt.show()
        pass

    def viewPcaSinglularValues(self, paramName):
        pca = load(self.protocol._getExtraPath('pca_pickled.pkl'))
        fig = plt.figure('PCA singlular values')
        plt.stem(pca.singular_values_)
        plt.xticks(np.arange(0, len(pca.singular_values_), 1))
        plt.show()
        pass

    def viewDendrogram(self, paramName):
        data = load(self.protocol._getExtraPath('covar_mat.pkl'))
        data = np.ones_like(data) - data
        plt.figure('Dendrogram')
        p = self.protocol.numOfClasses.get()
        dend = sch.dendrogram(sch.linkage(data, method='ward'), truncate_mode='lastp', p=p)
        # to show the whole dendrogram:
        # dend = sch.dendrogram(sch.linkage(data, method='ward'))
        plt.xlabel("# subtomograms")
        plt.ylabel('Distance "ward"')
        plt.title('Hierarchical clustering on 1 - $CCC_{ij}$')
        plt.show()
        pass

    def viewFullDendrogram(self, paramName):
        data = load(self.protocol._getExtraPath('covar_mat.pkl'))
        data = np.ones_like(data) - data
        plt.figure('Dendrogram')
        # show the whole dendrogram:
        dend = sch.dendrogram(sch.linkage(data, method='ward'))
        plt.xlabel("# subtomograms")
        plt.ylabel('Distance "ward"')
        plt.title('Hierarchical clustering on 1 - $CCC_{ij}$')
        plt.show()
        pass

