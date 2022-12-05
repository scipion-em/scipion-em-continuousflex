# **************************************************************************
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

import os
from os.path import basename, join, exists, isfile
import numpy as np
import pwem.emlib.metadata as md
from pyworkflow.utils.path import cleanPath, makePath
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam
from pwem.objects import SetOfParticles
from pyworkflow.gui.browser import FileBrowserWindow
from continuousflex.protocols.protocol_tomoflow_dimred import FlexProtDimredHeteroFlow
from continuousflex.protocols.data import Point, Data
from .plotter_vol import FlexNmaVolPlotter
from continuousflex.viewers.nma_vol_gui import ClusteringWindowVolHeteroFlow
from continuousflex.viewers.nma_vol_gui import TrajectoriesWindowVolHeteroFlow
from pwem.viewers.viewer_chimera import Chimera
from joblib import load, dump
from continuousflex.protocols.utilities.spider_files3 import open_volume, save_volume
import continuousflex
from continuousflex import Plugin
from subprocess import check_call
import sys
import matplotlib.pyplot as plt
from pwem.emlib.image import ImageHandler
from pyworkflow.protocol import params

FIGURE_LIMIT_NONE = 0
FIGURE_LIMITS = 1

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1
POINT_LIMITS_NONE = 0
POINT_LIMITS = 1


class FlexDimredHeteroFlowViewer(ProtocolViewer):
    """ Visualization of results from TomoFlow Dimred protocol
    """
    _label = 'viewer heteroflow dimred'
    _targets = [FlexProtDimredHeteroFlow]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayRawDeformation', StringParam, default='1 2',
                      label='Display raw deformation',
                      help='Type 1 to see the histogram of reduced dimensions, '
                           'using axis 1; \n '
                           'Type 2 to see the histogram of reduced dimensions, '
                           'using axis 2; etc. \n '
                           'Type 1 2 to see the histogram of reduced dimensions, using axes 1 and 2; \n'
                           'Type 1 2 3 to see the histogram of reduced dimensions, using axes 1, 2, '
                           'and 3; etc. '
                      )
        form.addParam('displayPcaSingularValues', LabelParam,
                      label="Display PCA singular values",
                      help="The values should help you see how many dimensions are in the data ")
        form.addParam('displayClustering', LabelParam,
                      label='Open clustering tool?',
                      help='Open a GUI to visualize the volumes as points '
                           'and select some of them to create new clusters, and compute 3D averages of the clusters')
        form.addParam('displayTrajectories', LabelParam,
                      label='Open trajectories tool?',
                      help='Open a GUI to visualize the volumes as points'
                           ' to draw and adjust trajectories.')
        form.addParam('graylevel',params.FloatParam, label='Gray-level threshold level for animations',
                      default=0.1, expertLevel=params.LEVEL_ADVANCED)
        form.addHidden('limits_modes', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually Use upper and lower values'],
                      default=FIGURE_LIMIT_NONE,
                      label='(1 - CC) limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='If you want to use a range of (1-CC) choose to set it manually.')
        form.addHidden('LimitLow', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Lower (1-CC) value',
                      help='The lower (1-CC) used in the graph')
        form.addHidden('LimitHigh', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Upper (1-CC) value',
                      help='The upper (1-CC) used in the graph')
        form.addParam('xlimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually x-axis limits'],
                      default=X_LIMITS_NONE,
                      label='x-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of x-axis limits')
        form.addParam('xlim_low', params.FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Lower x-axis limit')
        form.addParam('xlim_high', params.FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Upper x-axis limit')
        form.addParam('ylimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually y-axis limits'],
                      default=Y_LIMITS_NONE,
                      label='y-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of y-axis limits')
        form.addParam('ylim_low', params.FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Lower y-axis limit')
        form.addParam('ylim_high', params.FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Upper y-axis limit')
        form.addParam('zlimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually z-axis limits'],
                      default=Z_LIMITS_NONE,
                      label='z-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of z-axis limits')
        form.addParam('zlim_low', params.FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Lower z-axis limit')
        form.addParam('zlim_high', params.FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Upper z-axis limit')
        # Scatter points size and transparancy
        form.addParam('points_shades', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually point radius and transparancy'],
                      default=POINT_LIMITS_NONE,
                      label='Scatter points radius and transparancy', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use change the points radius and transparancy in the scatter plot'
                           '. By trying different values, it may help you discover the densest regions in the space.')
        line = form.addLine('Radius and transparancy',
                            condition='points_shades==%d' % POINT_LIMITS,
                            help='Values for points rarius have can be any positive real number.'
                                 ' Values for transparancy are between 0 and 1.')
        line.addParam('s', params.FloatParam, default=None, allowsNull=True,
                      label='Radius')
        line.addParam('alpha', params.FloatParam, default=None, allowsNull=True,
                        label='Transparancy')


    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayPcaSingularValues': self.viewPcaSinglularValues,
                'displayClustering': self._displayClustering,
                'displayTrajectories': self._displayTrajectories,
                }

    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)

    def _doViewRawDeformation(self, components):
        components = list(map(int, components.split()))
        dim = len(components)
        views = []

        if dim > 0:
            modeList = [m - 1 for m in components]
            modeNameList = ['Principle Component Axis %d' % m for m in components]
            missingList = []

            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)),
                                          title="Invalid input")]

            # Actually plot
            # plotter = FlexNmaVolPlotter(data=self.getData())
            if self.limits_modes == FIGURE_LIMIT_NONE:
                plotter = FlexNmaVolPlotter(data=self.getData(),
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                            s=self.s, alpha = self.alpha)
            else:
                plotter = FlexNmaVolPlotter(data=self.getData(),
                                            LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                            s=self.s, alpha = self.alpha)


            baseList = [basename(n) for n in modeNameList]

            self.getData().XIND = modeList[0]
            if dim == 1:
                plotter.plotArray1D("Histogram of: %s" % baseList[0],
                                    "Amplitude", "Number of volumes")
            else:
                self.getData().YIND = modeList[1]
                if dim == 2:
                    # plotter.plotArray2D("Reduced dimensions deformation amplitudes: %s vs %s" % tuple(baseList),
                    #                     *baseList)
                    plotter.plotArray2D_xy("%s vs %s" % tuple(baseList),
                                        *baseList)
                elif dim == 3:
                    self.getData().ZIND = modeList[2]
                    # plotter.plotArray3D("%s %s %s" % tuple(baseList),
                    #                     *baseList)
                    plotter.plotArray3D_xyz("%s %s %s" % tuple(baseList),
                                        *baseList)
            views.append(plotter)

        return views

    def _displayClustering(self, paramName):
        self.clusterWindow = self.tkWindow(ClusteringWindowVolHeteroFlow,
                                           title='Volume Clustering Tool',
                                           dim=self.protocol.reducedDim.get(),
                                           data=self.getData(),
                                           callback=self._createCluster,
                                           limits_mode=self.limits_modes,
                                           LimitL=self.LimitLow,
                                           LimitH=self.LimitHigh,
                                           xlim_low=self.xlim_low,
                                           xlim_high=self.xlim_high,
                                           ylim_low=self.ylim_low,
                                           ylim_high=self.ylim_high,
                                           zlim_low=self.zlim_low,
                                           zlim_high=self.zlim_high,
                                           s=self.s,
                                           alpha=self.alpha)
        return [self.clusterWindow]

    def _displayTrajectories(self, paramName):
        self.trajectoriesWindow = self.tkWindow(TrajectoriesWindowVolHeteroFlow,
                                                title='Trajectories Tool',
                                                dim=self.protocol.reducedDim.get(),
                                                data=self.getData(),
                                                callback=self._generateAnimation,
                                                loadCallback=self._loadAnimation,
                                                numberOfPoints=10,
                                                limits_mode=self.limits_modes,
                                                LimitL=self.LimitLow,
                                                LimitH=self.LimitHigh,
                                                xlim_low=self.xlim_low,
                                                xlim_high=self.xlim_high,
                                                ylim_low=self.ylim_low,
                                                ylim_high=self.ylim_high,
                                                zlim_low=self.zlim_low,
                                                zlim_high=self.zlim_high,
                                                s=self.s,
                                                alpha=self.alpha)

        return [self.trajectoriesWindow]

    def _createCluster(self):
        """ Create the cluster with the selected particles
        from the cluster. This method will be called when
        the button 'Create Cluster' is pressed.
        """
        # Write the particles
        prot = self.protocol
        project = prot.getProject()
        # inputSet = prot.getInputParticles().get()
        inputSet = prot.getInputParticles()
        makePath(prot._getTmpPath())
        fnSqlite = prot._getTmpPath('cluster_particles.sqlite')
        cleanPath(fnSqlite)
        partSet = SetOfParticles(filename=fnSqlite)
        partSet.copyInfo(inputSet)
        first = True
        for point in self.getData():
            if point.getState() == Point.SELECTED:
                particle = inputSet[point.getId()]
                partSet.append(particle)
        partSet.write()
        partSet.close()

        from continuousflex.protocols.protocol_batch_cluster_tomoflow import FlexBatchProtTomoFlowCluster

        newProt = project.newProtocol(FlexBatchProtTomoFlowCluster)
        clusterName = self.clusterWindow.getClusterName()
        if clusterName:
            newProt.setObjLabel(clusterName)
        newProt.inputHeteroFlowDimred.set(prot)
        newProt.sqliteFile.set(fnSqlite)
        project.launchProtocol(newProt)
        project.getRunsGraph()


    #TODO
    def _loadAnimationData(self, obj):
        pass

    def _loadAnimation(self):
        prot = self.protocol
        browser = FileBrowserWindow("Select the animation folder (animation_NAME)",
                                    self.getWindow(), prot._getExtraPath(),
                                    onSelect=self._loadAnimationData)
        browser.show()

    def _generateAnimation(self):
        prot = self.protocol
        # This is not getting the file correctly, we are workingaround it:
        # projectorFile = prot.getProjectorFile()
        projectorFile = prot._getExtraPath() + '/projector.txt'
        if isfile(projectorFile):
            print('Mapping found, the animation is exact inverse of the dimensionality reduction method')
        else:
            print('Mapping not found, the animation is an estimation of reversing the dimensionality reduction method')

        animation = self.trajectoriesWindow.getAnimationName()
        animationPath = prot._getExtraPath('animation_%s' % animation)

        cleanPath(animationPath)
        makePath(animationPath)
        animationRoot = join(animationPath, 'animation_%s' % animation)
        trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])

        if isfile(projectorFile):
            M = np.loadtxt(projectorFile)
            if prot.getMethodName()=='sklearn_PCA':
                pca = load(prot._getExtraPath('pca_pickled.txt'))
                deformations = pca.inverse_transform(trajectoryPoints)
            else:
                # TODO: add mean
                deformations = np.dot(trajectoryPoints, np.linalg.pinv(M))
            np.savetxt(animationRoot + 'trajectory.txt', trajectoryPoints)
        else:
            Y = np.loadtxt(prot.getOutputMatrixFile())
            X = np.loadtxt(prot.getDeformationFile())
            # Find closest points in deformations
            deformations = [X[np.argmin(np.sum((Y - p) ** 2, axis=1))] for p in trajectoryPoints]

        # get the original size of the input:
        mdImgs = md.MetaData(self.protocol.inputOpFlow.get()._getExtraPath('volumes_out.xmd'))
        N = 0
        for objId in mdImgs:
            N += 1

        # reading back all optical flows
        bigmat = []
        if(isfile(self.protocol._getExtraPath('bigmat_inverse.pkl'))):
            print('bigmat_inverse.txt found')
            # bigmat_pinv = np.loadtxt(self.protocol._getExtraPath('bigmat_inverse.txt'))
            bigmat_pinv = load(self.protocol._getExtraPath('bigmat_inverse.pkl'))
        else:
            if(isfile(self.protocol._getExtraPath('bigmat.pkl'))):
                bigmat = load(self.protocol._getExtraPath('bigmat.pkl'))
            else:
                for j in range(1, N+1):
                    flowj = self.read_optical_flow_by_number(j)
                    flowj = np.reshape(flowj, [3 * np.shape(flowj)[1] * np.shape(flowj)[2] * np.shape(flowj)[3]])
                    bigmat.append(flowj)
                bigmat = np.array(bigmat)
                print('bigmat created successfully')
                # np.savetxt(self.protocol._getExtraPath('bigmat.txt'),bigmat)
                dump(bigmat,self.protocol._getExtraPath('bigmat.pkl'))
                print('bigmat.pkl saved successfully')
            bigmat_pinv = np.linalg.pinv(bigmat)
            bigmat = None  # removing it from the memory
            # np.savetxt(self.protocol._getExtraPath('bigmat_inverse.txt'),bigmat_pinv)
            dump(bigmat_pinv,self.protocol._getExtraPath('bigmat_inverse.pkl'))

        line = np.matmul(bigmat_pinv, np.transpose(deformations))
        bigmat_pinv = None # removing if from the memory
        fnref = self.protocol._getExtraPath('reference.spi')
        shape = np.shape(open_volume(fnref))
        ref = open_volume(fnref)
        # dumping the reference volume to use it in an outside script
        # creating the directory Tmp since it is usually deleted after the execution of the protocol
        if not exists(self.protocol._getTmpPath()):
            os.mkdir(self.protocol._getTmpPath())
        ref_dump = self.protocol._getTmpPath('ref_dump.pkl')
        dump(ref, ref_dump)
        for i, trash in enumerate(deformations):
            flowi = np.transpose(line[:, i])
            flowi = np.reshape(flowi, [3, shape[0], shape[1], shape[2]])
            pathi = animationRoot + str(i).zfill(3) + 'deformed_by_opflow.vol'
            flow_dump = self.protocol._getTmpPath('flow_dump.pkl')
            dump(np.float32(flowi), flow_dump)
            args = " %s %s %s" % (ref_dump, flow_dump, pathi)
            script_path = continuousflex.__path__[0] + '/protocols/utilities/optflow_warp.py'
            command = "python " + script_path + args
            command = Plugin.getContinuousFlexCmd(command)
            check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                       env=None, cwd=None)

        fn_cxc = self.protocol._getExtraPath('chimera_%s.cxc' % animation)
        # cxc_command = 'open ' + animationPath + '/*.vol vseries true\n'
        cxc_command = 'open animation_%s/*.vol vseries true\n' % animation
        # cxc_command += 'volume #1 style surface level 8.0\n'
        cxc_command += 'volume #1 style surface level %f\n' % self.graylevel
        cxc_command += 'vseries play #1 loop true maxFrameRate 5 direction oscillate'
        with open(fn_cxc, 'w') as f:
            print(cxc_command, file=f)
        command = Chimera.getHome()+'/bin/ChimeraX ' + fn_cxc
        # print(Chimera.getHome())
        os.system(command)


    def loadData(self):
        """ Iterate over the images and their deformations
        to create a Data object with theirs Points.
        """
        # particles = self.protocol.getInputParticles().get()
        particles = self.protocol.getInputParticles()
        mat = np.loadtxt(self.protocol._getExtraPath('output_matrix.txt'))
        data = Data()
        for i, particle in enumerate(particles):
            pointData = mat[i,:]
            data.addPoint(Point(pointId=particle.getObjId(),
                                data=pointData,
                                weight=0))
        return data

    def _validate(self):
        errors = []
        return errors

    def read_optical_flow(self, path_flowx, path_flowy, path_flowz):
        # x = open_volume(path_flowx)
        x = ImageHandler().read(path_flowx).getData()
        # y = open_volume(path_flowy)
        y = ImageHandler().read(path_flowy).getData()
        # z = open_volume(path_flowz)
        z = ImageHandler().read(path_flowz).getData()

        l = np.shape(x)
        flow = np.zeros([3, l[0], l[1], l[2]])
        flow[0, :, :, :] = x
        flow[1, :, :, :] = y
        flow[2, :, :, :] = z
        return flow

    def read_optical_flow_by_number(self, num):
        op_path = self.protocol.inputOpFlow.get()._getExtraPath()+'/optical_flows/'
        path_flowx = op_path + str(num).zfill(6) + '_opflowx.spi'
        path_flowy = op_path + str(num).zfill(6) + '_opflowy.spi'
        path_flowz = op_path + str(num).zfill(6) + '_opflowz.spi'
        flow = self.read_optical_flow(path_flowx, path_flowy, path_flowz)
        return flow

    def viewPcaSinglularValues(self, paramName):
        pca = load(self.protocol._getExtraPath('pca_pickled.txt'))
        fig = plt.figure('PCA singlular values')
        plt.stem(pca.singular_values_)
        plt.xticks(np.arange(0, len(pca.singular_values_), 1))
        plt.show()
        pass
