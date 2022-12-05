# **************************************************************************
# * Authors:    Ilyes Hamitouche            (ilyes.hamitouche@upmc.fr)
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

from continuousflex.protocols.protocol_deep_hemnma_train import FlexProtDeepHEMNMATrain
from pyworkflow.protocol.params import LabelParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from subprocess import check_call
import sys
from continuousflex import Plugin


class FlexDeepHEMNMAViewer(ProtocolViewer):
    """ Visualization of results from the deepHEMNMA training protocol
    """
    _label = 'viewer deepHEMNMA'
    _targets = [FlexProtDeepHEMNMATrain]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displaycurves', LabelParam,
                      label="Display training curves",
                      help="Display the training and validation losses")


    def _getVisualizeDict(self):
        return {'displaycurves': self._viewcurves}


    def _viewcurves(self, pramName):
        import webbrowser
        logdir = self.protocol._getExtraPath('scalars/')
        command = "tensorboard --port=6006 --logdir " + logdir +'&'
        command = Plugin.getContinuousFlexCmd(command)
        check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr, env=None, cwd=None)
        webbrowser.open_new("http://localhost:6006/")
