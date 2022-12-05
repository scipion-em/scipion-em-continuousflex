# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@igbmc.fr)
# *             Slavica Jonic                (slavica.jonic@upmc.fr)
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

from os.path import basename
import tkinter as tk
import pyworkflow.gui as gui
from pyworkflow.gui.widgets import Button, HotButton
from continuousflex.protocols.data import Point
from . import PointSelectorVol
from continuousflex.viewers.plotter_vol import FlexNmaVolPlotter

FIGURE_LIMIT_NONE = 0
FIGURE_LIMITS = 1


class ClusteringWindowVol(gui.Window):
    """ This class creates a Window that will display some Point's
    contained in a Data object.
    It will allow to launch 1D, 2D and 3D plots by selecting any
    combination of the x1, x2...xn from the Point dimension.
    Points can be selected by either Click and Drag in the Scatter plot or..
    by creating an Expression.
    Finally, there is a button 'Create Cluster' that will call a callback 
    fuction to take care of it.
    """

    def __init__(self, **kwargs):
        gui.Window.__init__(self, minsize=(420, 200), **kwargs)

        self.dim = kwargs.get('dim')
        self.data = kwargs.get('data')
        self.callback = kwargs.get('callback', None)
        self.plotter = None

        # Adding figure limits option
        self.limits_modes = kwargs.get('limits_mode')
        self.LimitLow = kwargs.get('LimitL')
        self.LimitHigh = kwargs.get('LimitH')
        self.xlim_low = kwargs.get('xlim_low')
        self.xlim_high = kwargs.get('xlim_high')
        self.ylim_low = kwargs.get('ylim_low')
        self.ylim_high = kwargs.get('ylim_high')
        self.zlim_low = kwargs.get('zlim_low')
        self.zlim_high = kwargs.get('zlim_high')
        # Alpha and S are the transparancy and the size of the points, respectively
        self._alpha = kwargs.get('alpha')
        self._s = kwargs.get('s')

        content = tk.Frame(self.root)
        self._createContent(content)
        content.grid(row=0, column=0, sticky='news')
        content.columnconfigure(0, weight=1)
        # content.rowconfigure(1, weight=1)

    def _createContent(self, content):
        self._createFigureBox(content)
        self._createClusteringBox(content)
        self._updateSelectionLabel()

    def _addLabel(self, parent, text, r, c):
        label = tk.Label(parent, text=text, font=self.fontBold)
        label.grid(row=r, column=c, padx=5, pady=5, sticky='ne')
        return label

    def _createFigureBox(self, content):
        frame = tk.LabelFrame(content, text='Figure')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)  # , minsize=30)
        # Create the 'Axes' label
        self._addLabel(frame, 'Axes', 0, 0)

        # Create a listbox with x1, x2 ...
        listbox = tk.Listbox(frame, height=5,
                             selectmode=tk.MULTIPLE, bg='white')
        for x in range(1, self.dim + 1):
            listbox.insert(tk.END, 'x%d' % x)
        listbox.grid(row=0, column=1, padx=5, pady=5, sticky='nw')
        self.listbox = listbox

        # Selection controls
        self._addLabel(frame, 'Selection', 1, 0)
        # Selection label
        self.selectionVar = tk.StringVar()
        self.clusterLabel = tk.Label(frame, textvariable=self.selectionVar)
        self.clusterLabel.grid(row=1, column=1, sticky='nw', padx=5, pady=(10, 5))

        # --- Expression
        expressionFrame = tk.Frame(frame)
        expressionFrame.grid(row=2, column=1, sticky='news')
        tk.Label(expressionFrame, text='Expression').grid(row=0, column=0, sticky='ne')
        self.expressionVar = tk.StringVar()
        expressionEntry = tk.Entry(expressionFrame, textvariable=self.expressionVar,
                                   width=30, bg='white')
        expressionEntry.grid(row=0, column=1, sticky='nw')
        helpText = 'e.g. x1>0 and x1<100 or x3>20'
        tk.Label(expressionFrame, text=helpText).grid(row=1, column=1, sticky='nw')

        # Buttons    
        buttonFrame = tk.Frame(frame)
        buttonFrame.grid(row=5, column=1, sticky='sew', pady=(10, 5))
        buttonFrame.columnconfigure(0, weight=1)
        resetBtn = Button(buttonFrame, text='Reset', command=self._onResetClick)
        resetBtn.grid(row=0, column=0, sticky='ne', padx=(5, 0))
        updateBtn = Button(buttonFrame, text='Update Plot', imagePath='fa-refresh.png',
                           command=self._onUpdateClick)
        updateBtn.grid(row=0, column=1, sticky='ne', padx=5)

        frame.grid(row=0, column=0, sticky='new', padx=5, pady=(10, 5))

    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Cluster')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)  # , minsize=30)

        # Cluster line
        self._addLabel(frame, 'Cluster name', 0, 0)
        self.clusterVar = tk.StringVar()
        clusterEntry = tk.Entry(frame, textvariable=self.clusterVar,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=1, sticky='nw', pady=5)

        buttonsFrame = tk.Frame(frame, bg='green')
        buttonsFrame.grid(row=1, column=1,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)

        self.createBtn = HotButton(buttonsFrame, text='Create Cluster',
                                   tooltip="Select some points to create the cluster",
                                   imagePath='fa-plus-circle.png', command=self._onCreateClick)
        self.createBtn.grid(row=0, column=1)

        frame.grid(row=1, column=0, sticky='new', padx=5, pady=(5, 10))

    def _onResetClick(self, e=None):
        """ Clean the expression and the current selection. """
        self.expressionVar.set('')
        for point in self.data:
            point.setState(Point.NORMAL)
        self._onUpdateClick()

    def _onCreateClick(self, e=None):
        if self.callback:
            self.callback()

    def _evalExpression(self):
        """ Evaluate the input expression and add 
        matching points to the selection.
        """
        value = self.expressionVar.get().strip()
        if value:
            for point in self.data:
                if point.eval(value):
                    point.setState(Point.SELECTED)

    def _onUpdateClick(self, e=None):
        components = self.listbox.curselection()
        dim = len(components)

        if not dim:
            self.showWarning("Please select some Axis before updating plots.")
        else:
            modeList = components
            modeNameList = ['x%d' % (m + 1) for m in components]
            missingList = []

            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)),
                                          title="Invalid input")]

            if self.plotter is None or self.plotter.isClosed():
                # self.plotter = FlexNmaVolPlotter(data=self.data)
                # Actually plot
                if self.limits_modes == FIGURE_LIMIT_NONE:
                    self.plotter = FlexNmaVolPlotter(data=self.data,
                                                     xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                                     ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                                     zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                                     s=self._s, alpha=self._alpha)
                else:
                    self.plotter = FlexNmaVolPlotter(data=self.data,
                                                     LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                                     xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                                     ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                                     zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                                     s=self._s, alpha=self._alpha)
                doShow = True
            else:
                self.plotter.clear()
                doShow = False

            # Actually plot
            baseList = [basename(n) for n in modeNameList]

            self.data.XIND = modeList[0]

            if dim == 1:
                self.plotter.plotArray1D("Histogram for %s" % baseList[0],
                                         "Deformation value", "Number of volumes")
            else:
                self.data.YIND = modeList[1]
                if dim == 2:
                    self._evalExpression()
                    self._updateSelectionLabel()

                    ax = self.plotter.plotArray2D("Click and drag to add some points to the Cluster",
                                                   *baseList)
                    self.ps = PointSelectorVol(ax, self.data, callback=self._updateSelectionLabel,
                                               LimitL=self.LimitLow, LimitH=self.LimitHigh, alpha=self._alpha, s=self._s)
                elif dim == 3:
                    try:
                        del self.ps  # Remove PointSelector
                    except:
                        pass
                    self.data.ZIND = modeList[2]
                    self.plotter.plotArray3D("%s %s %s" % tuple(baseList), *baseList)

            if doShow:
                self.plotter.show()
            else:
                self.plotter.draw()

    def _updateSelectionLabel(self):
        selected = self.data.getSelectedSize()
        self.selectionVar.set('%d / %d points' % (selected, self.data.getSize()))

        if selected:
            self.createBtn.config(state=tk.NORMAL)
        else:
            self.createBtn.config(state=tk.DISABLED)

    def getClusterName(self):
        return self.clusterVar.get().strip()

    def _onClosing(self):
        if self.plotter:
            self.plotter.close()
        gui.Window._onClosing(self)


class ClusteringWindowVolHeteroFlow(ClusteringWindowVol):
    def _onUpdateClick(self, e=None):
        components = self.listbox.curselection()
        dim = len(components)

        if not dim:
            self.showWarning("Please select some Axis before updating plots.")
        else:
            modeList = components
            modeNameList = ['x%d' % (m + 1) for m in components]
            missingList = []

            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)),
                                          title="Invalid input")]

            if self.plotter is None or self.plotter.isClosed():
                # self.plotter = FlexNmaVolPlotter(data=self.data)
                # Actually plot
                if self.limits_modes == FIGURE_LIMIT_NONE:
                    self.plotter = FlexNmaVolPlotter(data=self.data,
                                                     xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                                     ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                                     zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                                     s=self._s, alpha=self._alpha)
                else:
                    self.plotter = FlexNmaVolPlotter(data=self.data,
                                                     LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                                     xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                                     ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                                     zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                                     s=self._s, alpha=self._alpha)
                doShow = True
            else:
                self.plotter.clear()
                doShow = False

            # Actually plot
            baseList = [basename(n) for n in modeNameList]

            self.data.XIND = modeList[0]

            if dim == 1:
                self.plotter.plotArray1D("Histogram for %s" % baseList[0],
                                         "Deformation value", "Number of volumes")
            else:
                self.data.YIND = modeList[1]
                if dim == 2:
                    self._evalExpression()
                    self._updateSelectionLabel()
                    # ax = self.plotter.createSubPlot("Click and drag to add some points to the Cluster",
                    #                                 *baseList)

                    ax = self.plotter.plotArray2D_xy("Click and drag to add some points to the Cluster",
                                                     *baseList)
                    self.ps = PointSelectorVol(ax, self.data, callback=self._updateSelectionLabel, alpha=self._alpha, s=self._s)
                    # self.ps = PointSelectorVol(ax, self.data, callback=None)
                elif dim == 3:
                    # del self.ps  # Remove PointSelector
                    try:
                        del self.ps  # Remove PointSelector
                    except:
                        pass
                    self.data.ZIND = modeList[2]
                    self.plotter.plotArray3D_xyz("%s %s %s" % tuple(baseList), *baseList)

            if doShow:
                self.plotter.show()
            else:
                self.plotter.draw()
