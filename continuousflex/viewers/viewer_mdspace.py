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
from continuousflex.protocols.protocol_mdspace import FlexProtMDSPACE
from continuousflex.viewers.viewer_genesis import FlexGenesisViewer
from continuousflex.protocols.utilities.genesis_utilities import *

from .plotter import FlexPlotter
from pwem.viewers import VmdView, ChimeraView
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
import pwem.emlib.metadata as md
import re

from matplotlib.pyplot import cm

class FlexMDSPACEViewer(FlexGenesisViewer):
    """ Visualization of results from the MDSPACE protocol
    """
    _label = 'MDSPACE Viewer'
    _targets = [FlexProtMDSPACE]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        FlexGenesisViewer._defineParams(self, form)
        group = form.addGroup('MDSPACE')
        group.addParam('displayPCA', params.LabelParam,
                      label='Display PCA space')
        group.addParam('pcaAxes', params.StringParam, default="1 2",
                       label='Axes to display' )
        group.addParam('displayFE', params.LabelParam,
                      label='Display free energy')
        group.addParam('feAxes', params.StringParam, default="1 2",
                       label='Axes to display' )
        group.addParam('freeEnergySize', params.IntParam, default=20,
                       label='Sampling size' )
    def _getVisualizeDict(self):
        dict = FlexGenesisViewer._getVisualizeDict(self)
        dict['displayPCA'] = self._plotPCA
        dict['displayFE'] = self._plotFE
        return dict

    def _plotPCA(self, p):
        axes_str = str.split(self.pcaAxes.get())
        axes = []
        for i in axes_str: axes.append(int(i.strip())-1)

        plotter = FlexPlotter(1,self.protocol.numberOfIter.get()
                              , figsize=(self.protocol.numberOfIter.get()*3,3))
        for i in range(self.protocol.numberOfIter.get()):
            self.protocol._iter= i
            pca = np.loadtxt(self.protocol.getPCAPrefix() + "_matrix.txt")[:,axes]
            ax = plotter.createSubPlot("PCA iter "+str(i+1), "component " + axes_str[0],
                                   "component " + axes_str[1], xpos=1, ypos=i+1)
            ax.scatter(pca[:,0],pca[:,1])
        plotter.show()
    def _plotFE(self, p):
        axes_str = str.split(self.feAxes.get())
        axes = []
        for i in axes_str: axes.append(int(i.strip())-1)
        size =self.freeEnergySize.get()

        plotter = FlexPlotter(1,self.protocol.numberOfIter.get()
                              , figsize=(self.protocol.numberOfIter.get()*3,3))
        for i in range(self.protocol.numberOfIter.get()):
            self.protocol._iter= i

            data = np.loadtxt(self.protocol.getPCAPrefix() + "_matrix.txt")[:,axes]
            xmin = np.min(data[:,0])
            xmax = np.max(data[:,0])
            ymin = np.min(data[:,1])
            ymax = np.max(data[:,1])
            x = np.linspace(xmin, xmax, size)
            y = np.linspace(ymin, ymax, size)
            count = np.zeros((size, size))
            for j in range(data.shape[0]):
                count[np.argmin(np.abs(x.T - data[j, 0])),
                      np.argmin(np.abs(y.T - data[j, 1]))] += 1
            img = -np.log(count / count.max())
            img[img == np.inf] = img[img != np.inf].max()

            xx, yy = np.mgrid[xmin:xmax:size * 1j, ymin:ymax:size * 1j]

            ax = plotter.createSubPlot("Free energy iter "+str(i+1), "component " + axes_str[0],
                                   "component " + axes_str[1], xpos=1, ypos=i+1)
            im = ax.contourf(xx, yy, img, cmap='jet')
            # im = ax.imshow(img.T[::-1, :],
            #                cmap="jet", interpolation="bicubic",
            #                extent=[xmin, xmax, ymin, ymax])
            cbar = plotter.figure.colorbar(im)
            cbar.set_label("$\Delta G / k_{B}T$")
        plotter.show()