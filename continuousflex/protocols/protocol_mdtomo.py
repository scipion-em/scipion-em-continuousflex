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

from continuousflex.protocols.protocol_genesis import FlexProtGenesis, EMFIT_VOLUMES, SIMULATION_NMMD

class FlexProtMDTOMO(FlexProtGenesis):
    """ Protocol to perform MDTOMO using GENESIS """
    _label = 'MDTOMO'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        FlexProtGenesis._defineParams(self, form)
        param = form.getParam("simulationType")
        param.setDefault(SIMULATION_NMMD)
        param = form.getParam("n_steps")
        param.setDefault(50000)
        param = form.getParam("EMfitChoice")
        param.setDefault(EMFIT_VOLUMES)
