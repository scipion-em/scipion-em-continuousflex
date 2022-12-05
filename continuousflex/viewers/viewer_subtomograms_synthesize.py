# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
# *           Rémi Vuillemot             (remi.vuillemot@upmc.fr)
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
from pwem.emlib import MetaData, MDL_ORDER
from pyworkflow.protocol.params import StringParam, LabelParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceExt
from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtSynthesizeSubtomo
import pwem.emlib.metadata as md
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt
from continuousflex.protocols.protocol_subtomograms_synthesize import NMA_YES


class FlexProtSynthesizeSubtomoViewer(ProtocolViewer):
    """ Visualization of results from synthesized subtomogrmas
    """
    _label = 'viewer synthetic subtomograms'
    _targets = [FlexProtSynthesizeSubtomo]
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
                      condition=self.protocol.confVar.get() == NMA_YES,
                      label='Display normal-mode amplitudes relationship',
                      help='Type 7 to see the histogram of amplitudes along mode 7; \n'
                           'type 8 to see the histogram of amplitudes along mode 8, etc.\n'
                           'Type 7 8 to see the 2D plot of amplitudes along modes 7 and 8.\n'
                           'Type 7 8 9 to see the 3D plot of amplitudes along modes 7, 8 and 9; etc.'
                           )
        form.addParam('displayHists', LabelParam,
                      label="Display shift and angle histograms",
                      help="Display shift and angle histograms")
        form.addParam('displayVolumes', LabelParam,
                      label="Display volumes",
                      help="Display the volumes that are generated")

    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayHists': self._viewHists,
                'displayVolumes': self._viewVolumes,
                } 
                        
    def _viewVolumes(self, paramName):
        volumes = self.protocol.outputVolumes
        return [ObjectView(self._project, volumes.strId(), volumes.getFileName())]

    def _viewHists(self, paramName):
        mdVolumes = md.MetaData(self.protocol._getExtraPath('GroundTruth.xmd'))
        X = []
        Y = []
        Z = []
        Rot = []
        Tilt= []
        Psi = []
        for objId in mdVolumes:
            X.append(mdVolumes.getValue(md.MDL_SHIFT_X,objId))
            Y.append(mdVolumes.getValue(md.MDL_SHIFT_Y, objId))
            Z.append(mdVolumes.getValue(md.MDL_SHIFT_Z, objId))
            Rot.append(mdVolumes.getValue(md.MDL_ANGLE_ROT, objId))
            Tilt.append(mdVolumes.getValue(md.MDL_ANGLE_TILT, objId))
            Psi.append(mdVolumes.getValue(md.MDL_ANGLE_PSI, objId))

        fig, ax = plt.subplots(2, 3)

        fig.suptitle('Histogram of generated rigid-body paramerers')
        ax[0,0].hist(X, bins=25)
        ax[0,0].set_title('Shift X')
        ax[0,1].hist(Y, bins=25)
        ax[0,1].set_title('Shift Y')
        ax[0,2].hist(Z, bins=25)
        ax[0,2].set_title('Shift Z')
        ax[1,0].hist(Rot, bins=25)
        ax[1,0].set_title('Rot')
        ax[1,1].hist(Tilt, bins=25)
        ax[1,1].set_title('Tilt')
        ax[1,2].hist(Psi, bins=25)
        ax[1,2].set_title('Psi')
        plt.show()

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
                modes_md = MetaData(replaceExt(self.protocol.inputModes.get().getFileName(),'xmd'))
                for i, objId in enumerate(modes_md):
                    modeId = modes_md.getValue(MDL_ORDER, objId)
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
            plotter = FlexNmaPlotter(data=self.getData())
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
        mdVolumes = md.MetaData(self.protocol._getExtraPath('GroundTruth.xmd'))
        data = Data()
        for objId in mdVolumes:
            pointData = list(mdVolumes.getValue(md.MDL_NMA,objId))
            # inserting 6 zeros for the first 6 never used modes
            for j in range(6):
                pointData.insert(0, 0)
            data.addPoint(Point(pointId=objId,
                                data=pointData,
                                weight=0.0))
        return data