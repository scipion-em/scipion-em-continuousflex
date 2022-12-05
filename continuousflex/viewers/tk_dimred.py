# **************************************************************************
# *
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
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

from continuousflex.viewers.nma_gui import TrajectoriesWindow, ClusteringWindow
import tkinter as tk
from pyworkflow.gui.widgets import Button, ComboBox
from tkinter import Radiobutton
import numpy as np
import scipy as sp
from continuousflex.protocols.data import Point
from sklearn.cluster import KMeans

TOOL_TRAJECTORY = 1
TOOL_CLUSTERING = 2


class PCAWindowDimred(TrajectoriesWindow, ClusteringWindow):

    def __init__(self, **kwargs):
        TrajectoriesWindow.__init__(self, **kwargs)
        self.saveClusterCallback = kwargs.get('saveClusterCallback', None)
        self.saveCallback = kwargs.get('saveCallback', None)
        self.numberOfPoints = kwargs.get('numberOfPoints', 10)

        self._alpha=self.alpha
        self._s=self.s
        self._clusterNumber = 0

    def _createContent(self, content):
        self._createModeBox(content)
        self._createFigureBox(content)
        self._createTrajectoriesBox(content)
        self._createClusteringBox(content)
        self._exportBox(content)

    def _createModeBox(self, content):
        frame = tk.LabelFrame(content, text='Interactive mode', font=self.fontBold)

        selFrame = tk.Frame(frame)
        self.selectTool = tk.IntVar()
        r1 = Radiobutton(selFrame, text="Trajectory Mode", variable=self.selectTool, value=TOOL_TRAJECTORY, command=self._onUpdateClick)
        r1.grid(row=0, column=0, padx=5)
        r2 = Radiobutton(selFrame, text="Selection Mode", variable=self.selectTool, value=TOOL_CLUSTERING, command=self._onUpdateClick)
        r2.grid(row=0, column=1, padx=5)
        self.selectTool.set(TOOL_TRAJECTORY)
        selFrame.grid(row=0, column=0)
        frame.grid(row=0, column=0, sticky='new', padx=5, pady=(10, 5))


    def _createFigureBox(self, content):
        frame = tk.LabelFrame(content, text='Figure', font=self.fontBold)
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)  # , minsize=30)
        # Create the 'Axes' label
        self._addLabel(frame, 'Axes', 0, 0)

        # Create a listbox with x1, x2 ...
        listbox = tk.Listbox(frame, height=5,
                             selectmode=tk.MULTIPLE, bg='white')
        for x in range(1, self.dim + 1):
            listbox.insert(tk.END, 'x%d' % x)
        listbox.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.listbox = listbox

        # Selection controls
        self._addLabel(frame, 'Rejection', 1, 0)
        # Selection label
        self.selectionVar = tk.StringVar()
        self.clusterLabel = tk.Label(frame, textvariable=self.selectionVar)
        self.clusterLabel.grid(row=1, column=1, sticky='w', padx=5, pady=(10, 5))
        self._updateSelectionLabel()
        # --- Expression
        expressionFrame = tk.Frame(frame)
        expressionFrame.grid(row=2, column=1, sticky='w')
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

        frame.grid(row=1, column=0, sticky='new', padx=5, pady=(10, 5))

    def _onUpdateClick(self,e=None):
        if self.selectTool.get() == TOOL_TRAJECTORY :
            TrajectoriesWindow._onUpdateClick(self,e)
            self.createClusterBtn.config(state=tk.DISABLED)
            if (self.pathData.getSize() < self.numberOfPoints):
                self.updateClusterBtn.config(state=tk.NORMAL)
            self.eraseBtn.config(state=tk.DISABLED)
            self.trajSimBtn.config(state=tk.NORMAL)


        if self.selectTool.get() == TOOL_CLUSTERING:
            ClusteringWindow._onUpdateClick(self, e)
            self.createClusterBtn.config(state=tk.NORMAL)
            self.updateClusterBtn.config(state=tk.DISABLED)
            self.trajSimBtn.config(state=tk.DISABLED)
            self.eraseBtn.config(state=tk.NORMAL)

    def _exportBox(self,content):
        frame = tk.LabelFrame(content, text='Import/Export', font=self.fontBold)

        nameFrame = tk.Frame(frame)
        nameFrame.grid(row=0, column=0, sticky='w', pady=(10, 5))

        label = tk.Label(nameFrame, text="Name", font=self.fontBold)
        label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.clusterName = tk.StringVar()
        clusterEntry = tk.Entry(nameFrame, textvariable=self.clusterName,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=2, pady=5)


        buttonFrame = tk.Frame(frame)
        buttonFrame.grid(row=1, column=0, sticky='w', pady=(10, 5))

        self.saveClusterBtn = Button(buttonFrame, text='Export to EM dataset', state=tk.DISABLED,
                              tooltip='export clusters to scipion', command=self._onSaveClusterClick)
        self.saveClusterBtn.grid(row=0, column=2, padx=5)

        self.saveBtn = Button(buttonFrame, text='Save animation state',
                              tooltip='Save the trajectory', command=self._onSaveClick)
        self.saveBtn.grid(row=0, column=3)

        self.loadBtn = Button(buttonFrame, text='Load animation state', imagePath='fa-folder-open.png',
                              tooltip='Load a previous PCA clustering', command=self._onLoadClick)
        self.loadBtn.grid(row=0, column=4)

        frame.grid(row=4, column=0, sticky='new', padx=5, pady=(10, 5))



    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Clustering', font=self.fontBold)
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)


        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=0, column=0,
                          sticky='new', padx=5, pady=5)

        label = tk.Label(buttonsFrame, text="From trajectory", font = self.fontItalic)
        label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.updateClusterBtn = Button(buttonsFrame, text='Cluster from traj', state=tk.DISABLED,
                                     tooltip='Create new cluster',
                                     imagePath='fa-plus-circle.png', command=self._onUpdateCluster)
        self.updateClusterBtn.grid(row=0, column=1, padx=5)


        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='new', padx=5, pady=5)

        label = tk.Label(buttonsFrame, text="From selection", font = self.fontItalic)
        label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.createClusterBtn = Button(buttonsFrame, text='New cluster from sel', state=tk.DISABLED,
                                     tooltip='New clutser from sel',
                                     imagePath='fa-plus-circle.png', command=self._onCreateCluster)
        self.createClusterBtn.grid(row=0, column=1, padx=5)

        self.eraseBtn = Button(buttonsFrame, text='Erase sel',  tooltip='Erase selection', command=self._onErase)
        self.eraseBtn.grid(row=0, column=2, padx=5)

        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=2, column=0,
                          sticky='new', padx=5, pady=5)

        label = tk.Label(buttonsFrame, text="K-means", font = self.fontItalic)
        label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        kmeansCluster = Button(buttonsFrame, text='Compute clusters', state=tk.NORMAL,
                                     tooltip='K means clustering', command=self._onKMeansCluster)
        kmeansCluster.grid(row=0, column=1, padx=5)

        label = tk.Label(buttonsFrame, text="Number of clusters")
        label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.numPointsKmean = tk.StringVar(value="3")
        clusterEntry = tk.Entry(buttonsFrame, textvariable=self.numPointsKmean ,
                                width=3, bg='white')
        clusterEntry.grid(row=0, column=3, pady=5)


        frame.grid(row=3, column=0, sticky='new', padx=5, pady=(10, 5))

    def _createTrajectoriesBox(self, content):
        frame = tk.LabelFrame(content, text='Trajectories', font=self.fontBold, highlightcolor="cyan")
        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=0, column=0,
                          sticky='w', padx=5, pady=5)

        label = tk.Label(buttonsFrame, text="Number of points")
        label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.numberOfPointsVar = tk.StringVar(value=str(self.numberOfPoints))
        nPointsEntry = tk.Entry(buttonsFrame, textvariable=self.numberOfPointsVar ,
                                width=3, bg='white')
        nPointsEntry.grid(row=0, column=1, pady=5)
        setNPointsBtn = Button(buttonsFrame, text='Set', state=tk.NORMAL,
                                tooltip='Set the number of points for the trajectory', command=self._onSetNPoints)
        setNPointsBtn.grid(row=0, column=2, padx=5)


        buttonsFrame2 = tk.Frame(frame)
        buttonsFrame2.grid(row=1, column=0,
                          sticky='w', padx=5, pady=5)
        buttonsFrame2.columnconfigure(0, weight=1)
        self.trajSimBtn = Button(buttonsFrame2, text='Generate points', state=tk.NORMAL,
                                     tooltip='Generate trajectory points based on axis and trajectory type', command=self._onSimClick)
        self.trajSimBtn.grid(row=0, column=0, padx=5)
        self.trajAxisBtn = ComboBox(buttonsFrame2, choices=["axis %i"%(i+1) for i in range(self.dim)])
        self.trajAxisBtn.grid(row=0, column=1, padx=(5, 10))
        self.trajTypeBtn = ComboBox(buttonsFrame2, choices=["percentiles",
                                                            "linear betmeen min and max", "Linear betmeen -2*std and +2*std"
                                                            , "Gaussian betmeen min and max", "Gaussian betmeen -2*std and +2*std"])
        self.trajTypeBtn.grid(row=0, column=2, padx=(5, 5))

        buttonsFrame3 = tk.Frame(frame)
        buttonsFrame3.grid(row=2, column=0,
                          sticky='w', padx=5, pady=5)
        buttonsFrame3.columnconfigure(0, weight=1)
        self.generateBtn = Button(buttonsFrame3, text='Show in VMD', state=tk.NORMAL,
                                     tooltip='Select trajectory points to generate the animations',
                                     imagePath='fa-plus-circle.png', command=self._onCreateClick)
        self.generateBtn.grid(row=0, column=0, padx=5)
        self.comboBtn = ComboBox(buttonsFrame3, choices=["Inverse transformation", "cluster average"])
        self.comboBtn.grid(row=0, column=1, padx=(5, 10))

        frame.grid(row=2, column=0, sticky='new', padx=5, pady=(5, 10))

    def _onSaveClusterClick(self, e=None):
        if self.saveClusterCallback:
            self.saveClusterCallback(self)

    def _onSaveClick(self, e=None):
        if self.saveCallback:
            self.saveCallback(self)

    def _onSetNPoints(self):
        try :
            self.numberOfPoints = int(self.numberOfPointsVar.get())
        except:
            self.showError("Can not read number of points.")

    def _onKMeansCluster(self):
        try :
            n_clusters = int(self.numPointsKmean.get())
        except:
            return self.showError("Can not read number of clusters")
        self._onUpdateClick()

        k_means = KMeans(init='k-means++', n_clusters=n_clusters)
        selection = np.array(self.listbox.curselection())
        data_arr = np.array([p.getData()[selection] for p in self.data])
        k_means.fit(data_arr)

        classes = k_means.labels_ + 1
        i=0
        for point in self.data:
            point._weight = classes[i]
            i+=1
        self.saveClusterBtn.config(state=tk.NORMAL)
        self._onUpdateClick()
        self.setClusterNumber(3)

    def _onSimClick(self, e=None):
        TrajectoriesWindow._onResetClick(self, e)
        traj_axis = self.trajAxisBtn.getValue()
        traj_type = self.trajTypeBtn.getValue()

        data_axis = np.array([p.getData()[traj_axis] for p in self.data])
        mean_axis =data_axis.mean()
        std_axis =data_axis.std()
        min_axis =data_axis.min()
        max_axis =data_axis.max()

        traj_points = np.zeros((self.numberOfPoints, self.dim))
        if traj_type== 0 :
            traj_points[:,traj_axis] = np.array(
                [np.percentile(data_axis,100*(i+1)/(self.numberOfPoints+1)) for i in range(self.numberOfPoints)]
            )
        elif traj_type== 1 :
            traj_points[:,traj_axis] = np.linspace(min_axis,max_axis,self.numberOfPoints)
        elif traj_type== 2:
            traj_points[:,traj_axis] = np.linspace(-2*std_axis,+2*std_axis,self.numberOfPoints)
        elif traj_type== 3:
            distribution = sp.stats.norm(loc=mean_axis, scale=std_axis)
            bounds_for_range = distribution.cdf([min_axis, max_axis])
            gaussTraj = distribution.ppf(np.linspace(*bounds_for_range, num=self.numberOfPoints))
            traj_points[:, traj_axis] = gaussTraj
        elif traj_type== 4:
            distribution = sp.stats.norm(loc=mean_axis, scale=std_axis)
            bounds_for_range = distribution.cdf([-2*std_axis, +2*std_axis])
            gaussTraj = distribution.ppf(np.linspace(*bounds_for_range, num=self.numberOfPoints))
            traj_points[:, traj_axis] = gaussTraj

        for i in range(self.numberOfPoints):
            self.pathData.addPoint(Point(pointId=i + 1, data=traj_points[i], weight=0))

        self._checkNumberOfPoints()
        self._onUpdateClick()

    def _onUpdateCluster(self):
        traj_arr = np.array([p.getData() for p in self.pathData])
        selection = np.array(self.listbox.curselection())
        traj_sel = traj_arr[:, selection]

        for point in self.data:
            point_sel = point.getData()[selection]
            closet_point = np.argmin(np.linalg.norm(traj_sel - point_sel, axis=1))
            point._weight = closet_point + 1

        self.saveClusterBtn.config(state=tk.NORMAL)
        self._onUpdateClick()
        self.setClusterNumber(self.numberOfPoints)


    def _onCreateCluster(self):
        self.setClusterNumber(self.getClusterNumber() +1)
        for point in self.data:
            if point.getState() == Point.SELECTED:
                point._weight =self.getClusterNumber()
        self.saveClusterBtn.config(state=tk.NORMAL)
        ClusteringWindow._onResetClick(self)

    def setClusterNumber(self, n):
        self._clusterNumber = n

    def _onErase(self):
        for point in self.data:
            if point.getState() == Point.SELECTED:
                point._weight =0.0
        self.saveClusterBtn.config(state=tk.NORMAL)
        ClusteringWindow._onResetClick(self)

    def _checkNumberOfPoints(self):
        TrajectoriesWindow._checkNumberOfPoints(self)
        self.updateClusterBtn.config(state=tk.NORMAL)

    def _onResetClick(self, e=None):
        self.updateClusterBtn.config(state=tk.DISABLED)
        self.saveClusterBtn.config(state=tk.DISABLED)
        self.setClusterNumber(0)

        for point in self.data:
            point._weight = 0
        TrajectoriesWindow._onResetClick(self, e)
        self.generateBtn.config(state=tk.NORMAL)


    def getClusterName(self):
        return self.clusterName.get().strip()

    def getAnimationType(self):
        return self.comboBtn.getValue()

    def getClusterNumber(self):
        return self._clusterNumber