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
from continuousflex.protocols.protocol_test import FlexProtTest

TYPE_1 = 0
TYPE_2 = 1
TYPE_3 = 2

class FlexTestViewer(ProtocolViewer):
    """ Visualization of results from the NMA alignment vol protocol (HEMNMA-3D)
    """
    _label = 'viewer testl'
    _targets = [FlexProtTest]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('choiceType', params.EnumParam,
                      choices=['type1', 'type2', "type3"],
                      default=TYPE_1,
                      label='choice type')

    def _getVisualizeDict(self):
        return {"choiceType": self.showType}

    def showType(self):
        pass

