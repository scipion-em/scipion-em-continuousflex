# **************************************************************************
# *
# * Authors:     Ilyes Hamitouche (ilyes.hamitouche@upmc.fr)
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
from pwem.emlib import MetaData, MDL_ORDER
from pyworkflow.protocol.params import StringParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol import params
from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtDeepHEMNMAInfer

FIGURE_LIMIT_NONE = 0
FIGURE_LIMITS = 1

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1


class FlexDeepHEMNMAinferViewer(ProtocolViewer):
    """ Visualization of results from the deepHEMNMA inference protocol
    """
    _label = 'viewer nma alignment'
    _targets = [FlexProtDeepHEMNMAInfer]
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
                      label='Display the computed normal-mode amplitudes',
                      help='Type 7 to see the histogram of amplitudes along mode 7; \n'
                           'type 8 to see the histogram of amplitudes along mode 8, etc.\n'
                           'Type 7 8 to see the 2D plot of amplitudes along modes 7 and 8.\n'
                           'Type 7 8 9 to see the 3D plot of amplitudes along modes 7, 8 and 9; etc.'
                      )
        form.addParam('limits_modes', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually Use upper and lower values'],
                      default=FIGURE_LIMIT_NONE,
                      label='Error limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='If you want to use a range of Error in the color bar choose to set it manually.')
        form.addParam('LimitLow', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Lower Error value',
                      help='The lower Error used in the graph')
        form.addParam('LimitHigh', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Upper Error value',
                      help='The upper Error used in the graph')
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

    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
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
                md = MetaData(self.protocol.trained_model.get().inputNMA.get()._getExtraPath('modes.xmd'))
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
                plotter = FlexNmaPlotter(data=self.getData(),
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high)
            else:
                plotter = FlexNmaPlotter(data=self.getData(),
                                            LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high)
            baseList = [basename(n) for n in modeNameList]

            self.getData().XIND = modeList[0]
            if dim == 1:
                plotter.plotArray1D("Histogram of normal-mode amplitudes: %s" % baseList[0],
                                    "Amplitude", "Number of images")
            else:
                self.getData().YIND = modeList[1]
                if dim == 2:
                    plotter.plotArray2D_xy("Normal-mode amplitudes: %s vs %s" % tuple(baseList), *baseList)
                elif dim == 3:
                    self.getData().ZIND = modeList[2]
                    plotter.plotArray3D_xyz("Normal-mode amplitudes: %s %s %s" % tuple(baseList), *baseList)
            views.append(plotter)

        return views

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
                                weight=0))
        return data