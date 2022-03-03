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

from os.path import basename
import os
from pyworkflow.utils import getListFromRangeString, removeBaseExt
from pwem.protocols import ProtAnalysis3D
from xmipp3.convert import writeSetOfVolumes
import pwem.emlib.metadata as md
from xmipp3 import XmippMdRow
from pyworkflow.utils.path import copyFile, makePath
import pyworkflow.protocol.params as params
from .convert import modeToRow
from pwem.utils import runProgram
from pwem import Domain
import glob
import time
from joblib import load, dump
import farneback3d
from continuousflex.protocols.utilities.spider_files3 import open_volume, save_volume
from continuousflex.protocols.utilities.src import Molecule, get_mol_conv, get_RMSD_coords
import numpy as np
from pwem.objects import SetOfAtomStructs, AtomStruct
import itertools
from pwem.convert.atom_struct import cifToPdb


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
        group.addParam('modeList', params.NumericRangeParam,
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
        form.addParam('applyMask', params.BooleanParam, label='Use a mask?', default=False,
                       help='A mask that can be applied to the reference without cropping it.'
                            ' A loose and smooth mask should be use.'
                            ' This mask is optional, we recommend to try without a mask and see'
                            ' if a mask improves the result later.'
                       )
        form.addParam('Mask', params.PointerParam,
                       condition='applyMask',
                       pointerClass='Volume', allowsNull=True,
                       label="Select mask")
        form.addParam('do_rmsd', params.BooleanParam, label= 'Compare the fitted structure with the groundtruth PDBs?',
                      expertLevel=params.LEVEL_ADVANCED, default= False)
        group = form.addGroup('Compare the fitting result with the groundtruth')
        group.addParam('targetPDBs', params.PathParam, allowsNull=True, condition='do_rmsd',
                      label='Target PDB(s)', help='This allows to show the accuracy of the method by passing the groundtruth'
                                                ' atomic structure that correspond to each input volume')
        form.addParam('fitIterations', params.IntParam, default=1,
                      label='number of iterations',
                      help = 'To do')
        form.addParam('regionSize', params.IntParam, default=3,
                      label='Region size (odd number)',
                      help = 'The optical flow region (voxels) around each atom that will be averaged')
        form.addParam('step_size', params.FloatParam, default=0.5,
                      label='Step size',
                      help='TO do... this will multiply the optical flow at each step to have smoother transition')
        form.addParam('do_HistEqual', params.BooleanParam, default=False,
                      label='Perform histogram matching?',
                      help='Histogram matching will be applied to the reference to match the gray level values of input volumes')
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
        group.addHidden('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='gray scale factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        group.addHidden('factor2', params.IntParam, default=100,
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

        # Make sure the region is an odd number:
        assert self.regionSize.get() % 2 == 1 , 'Only odd number for regions are allowed'
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
            dump(pdbs_list, self._getExtraPath('target_pdblist.pkl'))

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
            fit_directory = self._getExtraPath(removeBaseExt(basename(path_target)))
            makePath(fit_directory)
            if path_reference.endswith(".cif") or path_reference.endswith(".mmcif"):
                cifToPdb(path_reference, fit_directory + '/ref0.pdb')
            else:
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
                # Apply the mask on vol1
                if(self.applyMask.get()):
                    mask = open_volume(self.Mask.get().getFileName())
                    vol1 *= mask
                    save_volume(vol1, self._getExtraPath('masked_vol.vol'))
                # Apply histogram matching on vol0
                if(self.do_HistEqual.get()):
                    vol0 = match_histograms(vol0, vol1)

                #  Saving the cc, rmsd and normal mode amplitudes
                # objId has the number of volumes processed so far
                cc_a = np.array(cc)
                cc_a.resize(objId * n_loop)
                cc_a = cc_a.reshape(objId, -1)
                np.savetxt(self._getExtraPath('cc.txt'), cc_a)


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
                # multiply the optical flow by the step size:
                optFlow *= self.step_size.get()
                t_end = time.time()
                print("spent on calculating 3D optical flow", np.floor((t_end - t0) / 60), "minutes and",
                      np.round(t_end - t0 - np.floor((t_end - t0) / 60) * 60), "seconds")

                # Using the optical-flow to warp the reference to estimate the volumes:
                # TODO: save the optical flow of each iteration and the warped volume
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
                # optFlow shape is now (3, N, N, N) where N is the size of the volume
                optFlowAtom = np.zeros((init.n_atoms, 3))
                fitted = init.copy()
                origin = -np.ones(3) * size // 2

                # Combination of surrounding voxels
                set = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12]
                region_size = self.regionSize.get()
                set = set[:region_size+1]
                # The region is shifted one step to the positive since we do floor division for the coordinates
                comb = list(itertools.product(set, repeat=3))
                for i in range(init.n_atoms):
                    # interpolate optFlow for all atoms
                    coord = init.coords[i] / voxel_size - origin
                    floor = np.ndarray.astype(np.floor(coord), dtype=int)

                    optFlows = []
                    # TODO: make it faster using broadcasting instead of loop
                    for j in floor + comb:
                        if any(j < 0) or any(j >= optFlow.shape[1:]):
                            print("error volume and PDB not aligned :%s" % j)
                            print(" If the volume and PDB are aligned, this error could mean that either your volume is"
                                  " close to the boundaries and might need padding with zeros, or your region size"
                                  " is too big.")
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

                    rmsd_a = np.array(rmsd)
                    rmsd_a.resize(objId * n_loop)
                    # objId has the number of volumes
                    rmsd_a = rmsd_a.reshape((objId, -1))
                    np.savetxt(self._getExtraPath('rmsd.txt'), rmsd_a)

                if(self.NMA.get()):
                    nma_amplitudes_all.append(nma_amplitudes)

            # Saving the PDB with the highest cross correlation as final.pdb
            id = np.argmax(cc_a[objId-1,:])
            # print(id)
            copyFile(fit_directory + '/ref%i.pdb' % (id),fit_directory + '/final.pdb')
            mdPDBs.setValue(md.MDL_IMAGE, fit_directory + '/final.pdb', mdPDBs.addObject())
        mdPDBs.write(self._getExtraPath('PDBs.xmd'))


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