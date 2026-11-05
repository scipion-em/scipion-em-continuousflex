# ***************************************************************************
# * Authors:     Mohsen Kazemi (mkazemi@cnb.csic.es)
# *
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
# ***************************************************************************/
from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportPdb
from xmipp3.protocols import XmippProtConvertPdb
from continuousflex.protocols import FlexProtStructureMapping


class TestStructureMapping(BaseTest):
  @classmethod
  def setUpClass(cls):
    setupTestProject(cls)
    # open (4ake) and closed (1ake) adenylate kinase as clean volumes of the same size:
    # noisy subtomograms give thousands of pseudoatoms and the NMA then fails to converge
    cls.ds = DataSet.getDataSet('nma_V2.0')
    cls.protImport1 = cls.pdbToVolume('4ake_pdb')
    cls.protImport2 = cls.pdbToVolume('1ake_pdb')

  @classmethod
  def pdbToVolume(cls, key):
    protPdb = cls.newProtocol(ProtImportPdb, inputPdbData=1, pdbFile=cls.ds.getFile(key))
    cls.launchProtocol(protPdb)
    protVol = cls.newProtocol(XmippProtConvertPdb, sampling=2.2, setSize=True, size_x=64)
    protVol.pdbObj.set(protPdb.outputPdb)
    cls.launchProtocol(protVol)
    return protVol

  def testStructureMapping(self):
    """
        Test protocol structure mapping
    """
    protStrucMap = self.newProtocol(FlexProtStructureMapping,
                                    objLabel='structure mapping',
                                    numberOfMpi=1, numberOfThreads=4)
    protStrucMap.inputVolumes.append(self.protImport1.outputVolume)
    protStrucMap.inputVolumes.append(self.protImport2.outputVolume)
    protStrucMap.numberOfModes.set(20)
    protStrucMap.pseudoAtomTarget.set(2.0)
    protStrucMap.pseudoAtomRadius.set(0.75)
    protStrucMap.rcPercentage.set(90.0)
    protStrucMap.collectivityThreshold.set(0.2)
    self.launchProtocol(protStrucMap)