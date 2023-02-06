# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Slavica Jonic  (jonic@impmc.upmc.fr)
# *              Mohamad Harastani (mohamad.harastani@igbmc.fr)
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
# *
# **************************************************************************

import numpy as np
from .plotter import FlexPlotter

class FlexNmaPlotter(FlexPlotter):
    """ Add some extra plot utilities to XmippPlotter class, mainly for
    NMA vectors plotting of the deformations.txt file.
    """
    def __init__(self, **kwargs):
        """ Create the plotter, 'data' should be passed in **kwargs.
        """
        self._data = kwargs.get('data')
        self._limitlow = kwargs.get('LimitL')
        self._limitup = kwargs.get('LimitH')
        self._xlimlow = kwargs.get('xlim_low')
        self._xlimhigh = kwargs.get('xlim_high')
        self._ylimlow = kwargs.get('ylim_low')
        self._ylimhigh = kwargs.get('ylim_high')
        self._zlimlow = kwargs.get('zlim_low')
        self._zlimhigh = kwargs.get('zlim_high')
        # Alpha and S are the transparancy and the size of the points, respectively
        self._alpha = kwargs.get('alpha')
        self._cbar_label = kwargs.get('cbar_label', "Error")
        self._s = kwargs.get('s')
        FlexPlotter.__init__(self, **kwargs)
        self.useLastPlot = False
        
    def createSubPlot(self, title, xlabel, ylabel):
        if self.useLastPlot and self.last_subplot:
            ax = self.last_subplot
            ax.cla()
            ax.set_title(title)
        else:
            ax = FlexPlotter.createSubPlot(self, title, xlabel, ylabel)
        return ax
            
    def plotArray1D(self, title, xlabel, ylabel):
        xdata = self._data.getXData()
        ax = self.createSubPlot(title, xlabel, ylabel)
        ax.hist(xdata, 50)
        return ax
    
    def plotArray2D(self, title, xlabel, ylabel):
        ax = self.createSubPlot(title, xlabel, ylabel)
        lowx = lowy = None
        try:
            lowx = self._xlimlow.get()
            lowy = self._ylimlow.get()
        except:
            pass
        if lowx:
            ax.set_xlim([self._xlimlow.get(), self._xlimhigh.get()])
        if lowy:
            ax.set_ylim([self._ylimlow.get(), self._ylimhigh.get()])
        s = alpha= None
        try:
            s = self._s.get()
            alpha = self._alpha.get()
        except:
            pass
        if s and alpha:
            plotArray2D(ax, self._data, self._limitlow, self._limitup, s, alpha, cbar_label=self._cbar_label)
        else:
            plotArray2D(ax, self._data, self._limitlow, self._limitup, cbar_label=self._cbar_label)
        return ax

    def plotArray2D_xy(self, title, xlabel, ylabel):
        ax = self.createSubPlot(title, xlabel, ylabel)

        lowx = lowy = None
        try:
            lowx = self._xlimlow.get()
            lowy = self._ylimlow.get()
        except:
            pass

        if lowx:
            ax.set_xlim([self._xlimlow.get(), self._xlimhigh.get()])
        if lowy:
            ax.set_ylim([self._ylimlow.get(), self._ylimhigh.get()])
        s = alpha = None
        try:
            s = self._s.get()
            alpha = self._alpha.get()
        except:
            pass
        if s and alpha:
            plotArray2D_xy(ax, self._data, self._limitlow, self._limitup, s, alpha)
        else:
            plotArray2D_xy(ax, self._data, self._limitlow, self._limitup)

        return ax


    def plotArray3D(self, title, xlabel, ylabel, zlabel):
        import mpl_toolkits.mplot3d.axes3d as p3
        ax = p3.Axes3D(self.figure)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        xdata = self._data.getXData()
        ydata = self._data.getYData()
        zdata = self._data.getZData()
        weights = self._data.getWeights()

        lowx = lowy = lowz = None
        try:
            lowx = self._xlimlow.get()
            lowy = self._ylimlow.get()
            lowz = self._zlimlow.get()
        except:
            pass

        if lowx:
            ax.set_xlim([self._xlimlow.get(), self._xlimhigh.get()])
        if lowy:
            ax.set_ylim([self._ylimlow.get(), self._ylimhigh.get()])
        if lowz:
            ax.set_zlim([self._zlimlow.get(), self._zlimhigh.get()])

        color_low = color_high = None
        try:
            color_low = self._limitlow.get()
            color_high = self._limitup.get()
        except:
            pass

        alpha = 0.4
        s = 20
        try:
            alpha = self._alpha.get()
            s = self._s.get()
        except:
            pass
        if alpha is None or s is None:
            alpha = 0.4
            s = 20

        if color_low is None or color_high is None:
            cax = ax.scatter3D(xdata, ydata, zdata, c= weights, alpha=alpha, s=s)
        else:
            cax = ax.scatter3D(xdata, ydata, zdata, c= weights, vmin=color_low,
                               vmax=color_high, alpha=alpha, s=s)

        x2, y2, z2 = [], [], []
        for point in self._data:
            if point.getState() == 1:
                x2.append(point.getX())
                y2.append(point.getY())
                z2.append(point.getZ())
        ax.scatter(x2, y2, z2, color='yellow', alpha=0.4, s=8)
        cb = ax.figure.colorbar(cax)
        cb.set_label('Error')
        # Disable tight_layout that is not available for 3D
        self.tightLayoutOn = False
        return ax

    def plotArray3D_xyz(self, title, xlabel, ylabel, zlabel):
        import mpl_toolkits.mplot3d.axes3d as p3
        ax = p3.Axes3D(self.figure)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        xdata = self._data.getXData()
        ydata = self._data.getYData()
        zdata = self._data.getZData()
        weights = self._data.getWeights()

        lowx = lowy = lowz = None
        try:
            lowx = self._xlimlow.get()
            lowy = self._ylimlow.get()
            lowz = self._zlimlow.get()
        except:
            pass

        if lowx:
            ax.set_xlim([self._xlimlow.get(), self._xlimhigh.get()])
        if lowy:
            ax.set_ylim([self._ylimlow.get(), self._ylimhigh.get()])
        if lowz:
            ax.set_zlim([self._zlimlow.get(), self._zlimhigh.get()])

        alpha = 0.4
        s = 20
        try:
            alpha = self._alpha.get()
            s = self._s.get()
        except:
            pass
        if alpha is None or s is None:
            alpha = 0.4
            s = 20

        if self._limitlow == None or self._limitup == None:
            cax = ax.scatter3D(xdata, ydata, zdata, c= np.ones(len(weights))-weights, alpha=alpha, s=s)
        else:
            cax = ax.scatter3D(xdata, ydata, zdata, c= np.ones(len(weights))-weights, vmin=self._limitlow.get(),
                               vmax=self._limitup.get(), alpha=alpha, s=s)
        x2, y2, z2 = [], [], []
        for point in self._data:
            if point.getState() == 1:
                x2.append(point.getX())
                y2.append(point.getY())
                z2.append(point.getZ())
        # ax.scatter(x2, y2, z2, color='yellow', alpha=0.4, s=8)
        # cb = ax.figure.colorbar(cax)
        # cb.set_label('1- Cross Correlation')
        # Disable tight_layout that is not available for 3D
        self.tightLayoutOn = False
        return ax

#---------- Utility functions -----------------

def plotArray2D(ax, data, vvmin=None, vvmax=None, s = None, alpha = None, cbar_label=None):
    xdata = data.getXData()
    ydata = data.getYData()
    weights = data.getWeights()

    #Limit the number of points to 100000 otherwise, it slow down too much
    npoints = len(xdata)
    maxpoints = 100000
    if npoints > maxpoints :
        scale = npoints//maxpoints
        xdata = np.array(xdata)[::scale]
        ydata = np.array(ydata)[::scale]
        weights = np.array(weights)[::scale]
        
    if vvmin:
        cax = ax.scatter(xdata, ydata, c=weights, vmin=vvmin.get(), vmax=vvmax.get(), s=s, alpha=alpha)
    else:
        # cax = ax.scatter(xdata, ydata, weights)
        cax = ax.scatter(xdata, ydata, c=weights, s=s, alpha=alpha)
    cb = ax.figure.colorbar(cax)
    cb.set_label(cbar_label)

def plotArray2D_xy(ax, data, vvmin=None, vvmax=None, s = None, alpha = None):
    xdata = data.getXData()
    ydata = data.getYData()
    weights = data.getWeights()

    #Limit the number of points to 100000 otherwise, it slow down too much
    npoints = len(xdata)
    maxpoints = 100000
    if npoints > maxpoints :
        scale = npoints//maxpoints
        xdata = np.array(xdata)[::scale]
        ydata = np.array(ydata)[::scale]
        weights = np.array(weights)[::scale]

    if vvmin:
        cax = ax.scatter(xdata, ydata, c=weights, vmin=vvmin.get(), vmax=vvmax.get(), s=s, alpha=alpha)
    else:
        cax = ax.scatter(xdata, ydata, c=weights, s=s, alpha=alpha)