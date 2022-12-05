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

from os.path import basename
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LEVEL_ADVANCED
from pyworkflow.protocol import params
from continuousflex.protocols.protocol_nma_alignment_vol import FlexProtAlignmentNMAVol
from continuousflex.protocols.protocol_subtomograms_synthesize import FlexProtSynthesizeSubtomo
from continuousflex.protocols.data import Point, Data
from pwem.emlib import MetaData, MDL_ORDER, MDL_ANGLE_ROT, MDL_ANGLE_TILT, MDL_ANGLE_PSI, MDL_SHIFT_X, MDL_SHIFT_Y, \
    MDL_SHIFT_Z, MDL_NMA
from .plotter_vol import FlexNmaVolPlotter
from continuousflex.protocols.convert import l2
from xmippLib import SymList
import numpy as np
import tkinter.messagebox as mb
import matplotlib.pyplot as plt

FIGURE_LIMIT_NONE = 0
FIGURE_LIMITS = 1

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1

METADATA_PROJECT = 0
METADATA_FILE = 1


class FlexAlignmentNMAVolViewer(ProtocolViewer):
    """ Visualization of results from the NMA alignment vol protocol (HEMNMA-3D)
    """
    _label = 'viewer nma alignment vol'
    _targets = [FlexProtAlignmentNMAVol]
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
        form.addParam('displayRawDeformation', StringParam, default='7 8',
                      label='Display raw deformation',
                      help='Type 7 to see the histogram of raw deformation along mode 7; \n'
                           'type 8 to see the histogram of raw deformation along mode 8, etc.\n'
                           'Type 7 8 to see the 2D plot of raw deformations along modes 7 vs 8.\n'
                           'Type 7 8 9 to see the 3D plot of raw deformations along modes 7, 8 and 9; etc.'
                      )
        form.addParam('limits_modes', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually Use upper and lower values'],
                      default=FIGURE_LIMIT_NONE,
                      label='(1 - CC) limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='If you want to use a range of (1-CC) choose to set it manually.')
        form.addParam('LimitLow', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Lower (1-CC) value',
                      help='The lower (1-CC) used in the graph')
        form.addParam('LimitHigh', params.FloatParam, default=None,
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
        group = form.addGroup('Comparing with ground-truth', expertLevel=LEVEL_ADVANCED)
        group.addParam('GroundTruth', params.EnumParam,
                       choices=['From volume synthesis protocol', 'From an external metadata file'],
                       default=METADATA_PROJECT,
                       label='Ground-Truth parameters', display=params.EnumParam.DISPLAY_COMBO,
                       help='Use this is only when testing the method with synthetic data')
        group.addParam('SynthesisProject', params.PointerParam, pointerClass='FlexProtSynthesizeSubtomo',
                       condition='GroundTruth==%d' % METADATA_PROJECT,
                       allowsNull=True,
                       label="Project for volume synthesize",
                       help='Select a previous run for volume (subtomograms) synthesize.')
        group.addParam('MetadataFile', params.FileParam,
                       pointerClass='params.FileParam', allowsNull=True,
                       condition='GroundTruth==%d' % METADATA_FILE,
                       label="Metadata file (xmd)",
                       help='Choose a metadata file containing angles, shifts and NM amplitudes, typically a metadata'
                            ' file from synthesizing volumes')
        group.addParam('displayStatistics', params.LabelParam,
                       label="Display error statistics and plots?")

    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayStatistics': self._viewErrorStatistics,
                }


    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)

    def _doViewRawDeformation(self, components):
        components = list(map(int, components.split()))
        dim = len(components)
        views = []

        if dim > 0:
            modeList = []
            modeNameList = []
            missingList = []

            for modeNumber in components:
                found = False
                md = MetaData(self.protocol._getExtraPath('modes.xmd'))

                for i, objId in enumerate(md):
                    modeId = md.getValue(MDL_ORDER, objId)

                    if modeNumber == modeId:
                        modeNameList.append('Mode %d' % modeNumber)
                        modeList.append(i)
                        found = True
                        break
                if not found:
                    missingList.append(str(modeNumber))

            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)),
                                          title="Invalid input")]

            # Actually plot
            if self.limits_modes == FIGURE_LIMIT_NONE:
                plotter = FlexNmaVolPlotter(data=self.getData(),
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high)
            else:
                plotter = FlexNmaVolPlotter(data=self.getData(),
                                            LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high)

            baseList = [basename(n) for n in modeNameList]

            if dim == 1:
                self.getData().XIND = modeList[0]
                plotter.plotArray1D("Histogram for %s" % baseList[0],
                                    "Deformation Value", "Number of Volumes")
            else:
                self.getData().XIND = modeList[0]
                self.getData().YIND = modeList[1]
                if dim == 2:
                    plotter.plotArray2D("%s vs %s" % tuple(baseList), *baseList)
                elif dim == 3:
                    self.getData().ZIND = modeList[2]
                    plotter.plotArray3D("%s %s %s" % tuple(baseList), *baseList)
            views.append(plotter)

        return views

    def _viewErrorStatistics(self, paramName):
        if self.GroundTruth.get() == METADATA_PROJECT:
            metadata_file = self.SynthesisProject.get()._getExtraPath('GroundTruth.xmd')
        else:
            metadata_file = self.MetadataFile.get()
        return self._doViewErrorStatistics(metadata_file)

    def _doViewErrorStatistics(self, metadata_file):
        md_gt = MetaData(metadata_file)
        md_protocol = MetaData(self.protocol._getExtraPath('volumes.xmd'))
        md_protocol.sort()
        # Get the matching modes:
        md_modes = MetaData(self.protocol._getExtraPath('modes.xmd'))
        modeIds = []
        for i, objId in enumerate(md_modes):
            modeIds.append(md_modes.getValue(MDL_ORDER, objId))
        # Get the parameters from both lists:
        rtp_protocol = []
        xyz_protocol = []
        mode_ampl_protocol = []
        rtp_gt = []
        xyz_gt = []
        mode_ampl_gt = []
        for objId in md_protocol:
            rtp_protocol.append([md_protocol.getValue(MDL_ANGLE_ROT, objId),
                                 md_protocol.getValue(MDL_ANGLE_TILT, objId),
                                 md_protocol.getValue(MDL_ANGLE_PSI, objId)])
            xyz_protocol.append([md_protocol.getValue(MDL_SHIFT_X, objId),
                                 md_protocol.getValue(MDL_SHIFT_Y, objId),
                                 md_protocol.getValue(MDL_SHIFT_Z, objId)])
            mode_ampl_protocol.append(md_protocol.getValue(MDL_NMA, objId))

            rtp_gt.append([md_gt.getValue(MDL_ANGLE_ROT, objId),
                           md_gt.getValue(MDL_ANGLE_TILT, objId),
                           md_gt.getValue(MDL_ANGLE_PSI, objId)])
            xyz_gt.append([md_gt.getValue(MDL_SHIFT_X, objId),
                           md_gt.getValue(MDL_SHIFT_Y, objId),
                           md_gt.getValue(MDL_SHIFT_Z, objId)])
            mode_ampl_gt.append(md_gt.getValue(MDL_NMA, objId))

        # Angular and shift distances
        shift_distance = []
        angular_distance = []
        # The full description of computeDistanceAngles function is:
        # A = SymList.computeDistanceAngles(SymList(), rot1, tilt1, psi1, rot2, tilt2, psi2, projdir_mode, check_mirrors, object_rotation)
        # By default, they are all set to False. However, check_mirrors should be true in general.
        for i in range(len(rtp_protocol)):
            shift_distance.append(l2(xyz_gt[i], xyz_protocol[i]))
            angular_distance.append(SymList.computeDistanceAngles(SymList(),
                                                                  rtp_protocol[i][0], rtp_protocol[i][1], rtp_protocol[i][2],
                                                                  rtp_gt[i][0], rtp_gt[i][1], rtp_gt[i][2],
                                                                  False, True, False))
        # Normal mode amplitudes distances: we need to find the subset of normal modes used in alignment in the groundtruth
        counter = 0
        plt.figure()
        mean_amplitudes = []
        std_amplitudes = []
        label = []
        dist = []
        for i in modeIds:
            # mode 7 corresponds to zero in the ground truth, so we need to subtract 7
            A = np.array(mode_ampl_gt)[:,i - 7]
            B = np.array(mode_ampl_protocol)[:, counter]
            mean_amplitudes.append(np.mean(np.array(A - B)))
            std_amplitudes.append(np.std(np.array(A - B)))
            label.append('mode ' + str(i))
            dist.append(np.array(A - B))
            counter +=1
        plt.title('histogram of normal mode amplitude distances')
        plt.hist(dist, bins=100, label=label)
        plt.legend(loc='upper right')

        plt.figure()
        plt.hist(np.array(angular_distance), bins=100)
        plt.title('histogram of angular distance')
        plt.figure()
        plt.hist(np.array(shift_distance), bins=100)
        plt.title('histogram of shift distance')

        message = 'mean and standard deviation angular distance: ' + str(np.mean(np.array(angular_distance)))[:7]
        message += ' and ' + str(np.std(np.array(angular_distance)))[:7]
        message += '\nmean and standard deviation shift distance: ' + str(np.mean(np.array(shift_distance)))[:7]
        message += ' and ' + str(np.std(np.array(shift_distance)))[:7]

        counter = 0
        for i in modeIds:
            message += '\nmean and standard deviation for mode ' + str(i) + ': ' + str(mean_amplitudes[counter])[:7] + \
                       ' ' + str(std_amplitudes[counter])[:7]
            counter +=1

        mb.showinfo('Distances compared to the ground truth', message)
        plt.show()
        pass

    def loadData(self):
        """ Iterate over the images and their deformations 
        to create a Data object with theirs Points.
        """
        particles = self.protocol.outputParticles
        data = Data()
        for i, particle in enumerate(particles):
            pointData = list(map(float, particle._xmipp_nmaDisplacements))
            data.addPoint(Point(pointId=particle.getObjId(),
                                data=pointData,
                                weight=particle._xmipp_maxCC.get()))
        return data
