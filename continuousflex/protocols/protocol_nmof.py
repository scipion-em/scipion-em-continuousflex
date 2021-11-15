# **************************************************************************
# *
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
# *             Rémi Vuillemot               (remi.vuillemot@upmc.fr)
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

from os.path import basename, exists
import os
from pyworkflow.utils import getListFromRangeString
from pwem.protocols import ProtAnalysis3D
from xmipp3.convert import (writeSetOfVolumes, xmippToLocation, createItemMatrix,
                            setXmippAttributes, setOfParticlesToMd, getImageLocation)
import pwem as em
import pwem.emlib.metadata as md
from xmipp3 import XmippMdRow
from pyworkflow.utils.path import copyFile, cleanPath, makePath
import pyworkflow.protocol.params as params
from pyworkflow.protocol.params import NumericRangeParam
from .convert import modeToRow
from pwem.convert.atom_struct import cifToPdb
from pyworkflow.utils import removeBaseExt
from pwem.utils import runProgram
from pwem import Domain
import glob
import time
from joblib import load, dump
import farneback3d
from continuousflex.protocols.utilities.spider_files3 import open_volume, save_volume
from continuousflex.protocols.utilities.src import Molecule
from continuousflex.protocols.utilities.src import get_mol_conv, get_RMSD_coords
import numpy as np
from pwem.objects import SetOfAtomStructs, AtomStruct



class FlexProtNMOF(ProtAnalysis3D):
    """ Protocol for fitting using normal mode analysis and 3D dense optical flow. """
    _label = 'nmof'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('NMA', params.BooleanParam, label='Use normal modes and optical flows?',
                      default=True)
        group = form.addGroup('Input normal modes',
                               condition='NMA')
        group.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes", allowsNull=True,
                      help='Set of modes computed by normal mode analysis.')
        group.addParam('modeList', NumericRangeParam,
                      label="Modes selection (optional)",
                      help='Select the normal modes that will be used for volume analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for image analysis.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        group = form.addGroup('Input PDB (using optical flow fitting only)',
                              condition='NMA is False')
        group.addParam('pdb', params.PointerParam, label='input (pseudo)atomic structure',
                       pointerClass='AtomStruct', allowsNull=True,
                       help='The input structure can be an atomic model '
                            '(true PDB) or a pseudoatomic model\n'
                            '(an EM volume converted into pseudoatoms)')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select a volume or a set of volumes that will be fitted using the method')
        form.addParam('do_rmsd', params.BooleanParam, label= 'Compare the fitted structure with the groundtruth PDBs?',
                      expertLevel=params.LEVEL_ADVANCED, default= False)
        group = form.addGroup('Compare the fitting result with the groundtruth')
        group.addParam('targetPDBs', params.PathParam, allowsNull=True, condition='do_rmsd',
                      label='Target PDB(s)', help='This allows to show the accuracy of the method by passing the groundtruth'
                                                ' atomic structure that correspond to each input volume')
        form.addParam('fitIterations', params.IntParam, default=1,
                      label='number of iterations',
                      help = 'To do')
        # TODO: add mask
        # TODO: Histogram equalization
        # form.addParam('do_HistEqual', params.BooleanParam, default=True,
        #               label='Perform histogram equalization on the set of volumes?')
        form.addParam('do_rigidbody', params.BooleanParam, default=True,
                      label='Perform rigid-body alignment on the volume(s)?')
        group = form.addGroup('rigid-body alignment settings (Fast rotational matching)',
                              condition='do_rigidbody')
        group.addParam('frm_freq', params.FloatParam, default=0.25,
                      label='Maximum cross correlation frequency',
                      help='The normalized frequency should be between 0 and 0.5 '
                           'The more it is, the bigger the search frequency is, the more time it demands, '
                           'keeping it as default is recommended.')
        group.addParam('frm_maxshift', params.IntParam, default=10,
                      label='Maximum shift for rigid body alignment (in pixels)',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'It represents the maximum distance searched in x,y and z directions. Keep as default'
                           ' if your target is near the center in your subtomograms')
        form.addSection(label='3D optical flow parameters')
        group = form.addGroup('Optical flows')
        # TODO: allow multi processing of volumes on GPUs
        # group.addParam('N_GPU', params.IntParam, default=3, important=True, allowsNull=True,
        #                       label = 'Parallel processes on GPU',
        #                       help='This parameter indicates the number of volumes that will be processed in parallel'
        #                            ' (independently). The more powerful your GPU, the higher the number you can choose.')
        group.addParam('pyr_scale', params.FloatParam, default=0.5,
                      label='pyr_scale', allowsNull=True,
                       help='parameter specifying the image scale to build pyramids for each image (scale < 1).'
                            ' A classic pyramid is of generally 0.5 scale, every new layer added, it is'
                            ' halved to the previous one.')
        group.addParam('levels', params.IntParam, default=4, allowsNull=True,
                      label='levels',
                      help='evels=1 says, there are no extra layers (only the initial image).'
                           ' It is the number of pyramid layers including the first image.')
        group.addParam('winsize', params.IntParam, default=10, allowsNull=True,
                      label='winsize',
                      help='It is the average window size, larger the size, the more robust the algorithm is to noise,'
                           ' and provide smaller conformation detection, though gives blurred motion fields.'
                           ' You may try smaller window size for larger conformations but the method will be'
                           ' more sensitive to noise.')
        group.addParam('num_iterations', params.IntParam, default=10, allowsNull=True,
                      label='iterations',
                      help='Number of iterations to be performed at each pyramid level.')
        group.addParam('poly_n', params.IntParam, default=5, allowsNull=True,
                      label='poly_n',
                      help='It is typically 5 or 7, it is the size of the pixel neighbourhood which is used'
                           ' to find polynomial expansion between the pixels.')
        group.addParam('poly_sigma', params.FloatParam, default=1.2,
                      label='poly_sigma',
                      help='standard deviation of the gaussian that is for derivatives to be smooth as the basis of'
                           ' the polynomial expansion. It can be 1.2 for poly= 5 and 1.5 for poly= 7.')
        # TODO: histogram equalization should replace the need of two factors (one would be enough, and hidden)
        group.addParam('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='gray scale factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        group.addParam('factor2', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='gray scale factor2',
                      help='this factor will be multiplied by the gray levels of the reference')

        form.addParallelSection(threads=0, mpi=5)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        self.imgsFn_backup = self._getExtraPath('volumes_backup.xmd')
        self.modesFn = self._getExtraPath('modes.xmd')
        # Copy the input PDB to self.atomsFn
        if(self.NMA.get()):
            atomsFn = self.getInputPdb().getFileName()
            self.structureEM = self.inputModes.get().getPdb().getPseudoAtoms()
            if self.structureEM:
                self.atomsFn = self._getExtraPath(basename(atomsFn))
                copyFile(atomsFn, self.atomsFn)
            else:
                pdb_name = os.path.dirname(self.inputModes.get().getFileName()) + '/atoms.pdb'
                self.atomsFn = self._getExtraPath(basename(pdb_name))
                copyFile(pdb_name, self.atomsFn)
        else:
            atomsFn = self.pdb.get().getFileName()
            self.atomsFn = self._getExtraPath(basename(atomsFn))
            copyFile(atomsFn, self.atomsFn)

        # Write a metadata file input volumes and maybe modes
        self._insertFunctionStep('convertInputStep')

        # Align volumes if needed
        if(self.do_rigidbody):
            self._insertFunctionStep('alignInputVolumes')

        self._insertFunctionStep("performNMOF")
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        if(self.NMA.get()):
            # Write the modes metadata taking into account the selection
            self.writeSelectedModesMetaData()
        # Write metadata file with input volumes
        try:
            # If we have a set of volumes
            writeSetOfVolumes(self.inputVolumes.get(), self.imgsFn)
        except:
            # If one volume
            mdF = md.MetaData()
            mdF.setValue(md.MDL_IMAGE, self.inputVolumes.get().getFileName(), mdF.addObject())
            mdF.write(self.imgsFn)


    def writeSelectedModesMetaData(self):
        """ Iterate over the input SetOfNormalModes and write
        the proper Xmipp metadata.
        Take into account a possible selection of modes
        """
        if self.modeList.empty():
            modeSelection = []
        else:
            modeSelection = getListFromRangeString(self.modeList.get())

        mdModes = md.MetaData()

        inputModes = self.inputModes.get()
        for mode in inputModes:
            # If there is a mode selection, only
            # take into account those selected
            if not modeSelection or mode.getObjId() in modeSelection:
                row = XmippMdRow()
                modeToRow(mode, row)
                row.writeToMd(mdModes, mdModes.addObject())
        mdModes.write(self.modesFn)

    def alignInputVolumes(self):
        # Generate a volume from the input PDB:
        atomFn = self.atomsFn
        reference = self._getExtraPath('ref')
        sampling = self.inputVolumes.get().getSamplingRate()
        size = self.inputVolumes.get().getXDim()
        args = "-i %(atomFn)s -o %(reference)s --sampling %(sampling)s --size %(size)s --centerPDB -v 0" % locals()
        runProgram('xmipp_volume_from_pdb', args)
        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()
        tempdir = self._getTmpPath('')
        # Perform the alignment using MPI
        imgFn = self.imgsFn
        imgFnAligned = self._getExtraPath('aligned.xmd')
        # The stupid xmipp program adds .vol by default
        reference = self._getExtraPath('ref.vol')
        args = "-i %(imgFn)s -o %(imgFnAligned)s --odir %(tempdir)s --resume --ref %(reference)s" \
               " --frm_parameters %(frm_freq)f %(frm_maxshift)d " % locals()
        # print('xmipp_volumeset_align ', args)
        # This is needed to run MPI
        self.runJob("xmipp_volumeset_align", args,
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())
        # apply the alignment
        alignedPath = self._getExtraPath('aligned/')
        makePath(alignedPath)
        mdImgs = md.MetaData(imgFnAligned)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            rot = mdImgs.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdImgs.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdImgs.getValue(md.MDL_ANGLE_PSI, objId)
            x_shift = mdImgs.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdImgs.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdImgs.getValue(md.MDL_SHIFT_Z, objId)

            new_imgPath = alignedPath+basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)

            params = '-i %(imgPath)s -o %(new_imgPath)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                     ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()
            runProgram('xmipp_transform_geometry', params)

        # Sorting after alignment is importatnt if target PDBs are passed
        mdImgs.sort()
        mdImgs.write(imgFnAligned)
        # Point to the aligned volumes as input
        self.imgsFn = imgFnAligned


    def performNMOF(self):
        # if target PDBs are passed, get them and sort them
        pdbs_list = []
        if(self.do_rmsd.get()):
            pdbs_list = [f for f in glob.glob(self.targetPDBs.get())]
            pdbs_list.sort()
        # Loop over the volumes:
        mdImgs = md.MetaData(self.imgsFn)
        rmsd = []
        cc = []
        nma_amplitudes_all = []
        mdPDBs = md.MetaData()
        for objId in mdImgs:
            path_reference = self.atomsFn
            path_target = mdImgs.getValue(md.MDL_IMAGE, objId)
            if(self.do_rmsd.get()):
                path_target_pdb = pdbs_list[objId-1]
            voxel_size = self.inputVolumes.get().getSamplingRate()
            size = self.inputVolumes.get().getXDim()
            n_loop = self.fitIterations.get()
            # Todo: make a directory for each input volume to place the fitting results
            fit_directory = self._getExtraPath(removeBaseExt(basename(path_target)))
            makePath(fit_directory)
            copyFile(path_reference, fit_directory + '/ref0.pdb' )
            # Loop for each volume and iteratively approach its conformation
            nma_amplitudes = []
            for n in range(n_loop):
                print("Loop #%i" % n)
                # Transform to vol
                structure_i = fit_directory + '/ref%i.pdb' %n
                volume_i = fit_directory + '/ref%i' %n
                args = '-i %(structure_i)s -o %(volume_i)s --sampling %(voxel_size)s --centerPDB' \
                       ' --size %(size)s -v 0' %locals()
                runProgram('xmipp_volume_from_pdb', args)
                # Find and store the cross correlation at each iteration:



                # stupid xmipp program adds .vol to any volume name
                volume_i = fit_directory + '/ref%i.vol' % n
                vol0 = open_volume(volume_i)
                vol1 = open_volume(path_target)
                cc.append(CC(vol0, vol1))
                # numerical gray-level adjustments
                # TODO: replace with one factor if histogram equalization is done
                vol0 = vol0 * self.factor1.get()
                vol1 = vol1 * self.factor2.get()

                # optical flow calculation
                optflow = farneback3d.Farneback(
                    pyr_scale=self.pyr_scale.get(), # Scaling between multi-scale pyramid levels
                    levels=self.levels.get(),  # Number of multi-scale levels
                    winsize=self.winsize.get(),  # Window size for Gaussian filtering of polynomial coefficients
                    num_iterations=self.num_iterations.get(),  # Iterations on each multi-scale level
                    poly_n=self.poly_n.get(),  # Size of window for weighted least-square estimation of polynomial coefficients
                    poly_sigma=self.poly_sigma.get(),  # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
                )

                t0 = time.time()
                # perform OF:
                optFlow = optflow.calc_flow(vol0, vol1)
                t_end = time.time()
                print("spent on calculating 3D optical flow", np.floor((t_end - t0) / 60), "minutes and",
                      np.round(t_end - t0 - np.floor((t_end - t0) / 60) * 60), "seconds")

                # Using the optical-flow to warp the reference to estimate the volumes:
                volume_i = fit_directory + '/ref%i.vol' % n
                warped_i = fit_directory + '/warped.vol'
                vol0 = open_volume(volume_i)
                warped = farneback3d.warp_by_flow(vol0, optFlow)
                save_volume(warped, warped_i)

                # Pickiling the optical flow
                dump(optFlow, fit_directory + '/optical_flow.pkl')

                # print("> Apply optical flow to PDB ...")
                # Reading Molecule
                init = Molecule(structure_i)
                init.center()

                # preparing variables
                optFlow = np.transpose(optFlow, (0, 3, 2, 1))
                optFlowAtom = np.zeros((init.n_atoms, 3))
                fitted = init.copy()
                origin = -np.ones(3) * size // 2

                for i in range(init.n_atoms):
                    # interpolate optFlow for all atoms
                    coord = init.coords[i] / voxel_size - origin
                    floor = np.ndarray.astype(np.floor(coord), dtype=int)
                    comb = np.array(
                        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
                    optFlows = []
                    for j in floor + comb:
                        if any(j < 0) or any(j >= optFlow.shape[1:]):
                            print("error volume and PDB not aligned :%s" % j)
                        optFlows.append(optFlow[:, j[0], j[1], j[2]])
                    optFlowAtom[i] = np.mean(optFlows, axis=0)

                    # Apply transformation to atoms
                    fitted.coords[i] = init.coords[i] + -optFlowAtom[i]

                # Save the new PDB
                fitted.save_pdb(fit_directory + '/ref%i.pdb' % (n + 1))

                # If NMA is used, then project the optical flow on the selected normal modes then use the resultant
                # amplidutes to generate a fitted structure
                amplitudes = []
                if(self.NMA.get()):
                    # the structure with normal modes
                    ref_path = fit_directory + '/ref0.pdb'
                    # This will be overwritten
                    fitted_path = fit_directory + '/ref%i.pdb' % (n + 1)
                    fit_directory + '/ref%i.pdb' % (n + 1)
                    ref_nma = Molecule(ref_path)
                    ref_nma.center()
                    diff = np.array(fitted.coords - ref_nma.coords)
                    modes = readModes(self.modesFn)
                    for mode in modes:
                        amplitudes.append(np.dot(diff.flatten(), mode.flatten()))
                    # Save the amplitudes for comparing later
                    nma_amplitudes.append(amplitudes)
                    amplitudes = ' '.join(map(str, amplitudes))
                    modesFn = self.modesFn
                    args = '--pdb %(ref_path)s --nma %(modesFn)s --deformations' \
                           ' %(amplitudes)s -o %(fitted_path)s -v 0' % locals()
                    runProgram('xmipp_pdb_nma_deform', args)
                    # Read the updated fitted structure (in case rmsd will be found)
                    fitted = Molecule(fitted_path)
                    fitted.center()



                if(n == 0):
                    if(self.do_rmsd.get()):
                        # Add the rmsd with the initial structure
                        target_pdb = Molecule(path_target_pdb)
                        target_pdb.center()
                        idx = get_mol_conv(init, target_pdb)
                        rmsd_i = get_RMSD_coords(init.coords[idx[:, 0]], target_pdb.coords[idx[:, 1]])
                        rmsd.append(rmsd_i)

                if (self.do_rmsd.get()):
                    # Compute RMSD
                    target_pdb = Molecule(path_target_pdb)
                    target_pdb.center()
                    idx = get_mol_conv(fitted, target_pdb)
                    rmsd_i = get_RMSD_coords(fitted.coords[idx[:, 0]], target_pdb.coords[idx[:, 1]])
                    rmsd.append(rmsd_i)

                if(self.NMA.get()):
                    nma_amplitudes_all.append(nma_amplitudes)

            fitted.save_pdb(fit_directory + '/final.pdb')
            mdPDBs.setValue(md.MDL_IMAGE, fit_directory + '/final.pdb', mdPDBs.addObject())
        mdPDBs.write(self._getExtraPath('PDBs.xmd'))


        #  Saving the cc, rmsd and normal mode amplitudes
        # objId has the number of volumes
        cc = np.array(cc)
        cc = cc.reshape(objId, -1)
        np.savetxt(self._getExtraPath('cc.txt'), cc)

        if rmsd == []:
            pass
        else:
            rmsd = np.array(rmsd)
            # objId has the number of volumes
            rmsd = rmsd.reshape((objId,-1))
            np.savetxt(self._getExtraPath('rmsd.txt'), rmsd)
        print(rmsd)

        if nma_amplitudes == []:
            pass
        else:
            nma_amplitudes_all = np.array(nma_amplitudes_all)
            dump(nma_amplitudes_all,self._getExtraPath('nma_amplitudes_all.pkl'))


    def createOutputStep(self):
        fnSqlite = self._getPath('atomic_structures.sqlite')
        PDBs = SetOfAtomStructs(filename=fnSqlite)
        md_PDBs =  md.MetaData(self._getExtraPath('PDBs.xmd'))
        for objId in md_PDBs:
            fn_pdb = md_PDBs.getValue(md.MDL_IMAGE, objId)
            pdb_item = AtomStruct(filename=fn_pdb)
            PDBs.append(pdb_item)
        PDBs.write()
        self._defineOutputs(alignedPDBs=PDBs)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2020hybrid','Jonic2005', 'Sorzano2004b', 'Jin2014']

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    def _getLocalModesFn(self):
        modesFn = self.inputModes.get().getFileName()
        return self._getBasePath(modesFn)

    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
                           md.MDL_SHIFT_Y, md.MDL_SHIFT_Z, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST, md.MDL_MAXCC,
                           md.MDL_ANGLE_Y)
        createItemMatrix(item, row, align=em.ALIGN_PROJ)

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """

        return self.inputModes.get().getPdb()

def readModes(fnIn):
    modesMD = md.MetaData(fnIn)
    vectors = []
    for objId in modesMD:
        vecFn = modesMD.getValue(md.MDL_NMA_MODEFILE, objId)
        vec = np.loadtxt(vecFn)
        vectors.append(vec)
    return vectors

def CC(map1, map2):
    return np.sum(map1*map2)/np.sqrt(np.sum(np.square(map1))*np.sum(np.square(map2)))