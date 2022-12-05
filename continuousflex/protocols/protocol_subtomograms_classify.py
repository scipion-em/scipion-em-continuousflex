# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@igbmc.fr)
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

from pyworkflow.object import String
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils.path import makePath, copyFile, removeBaseExt
from pyworkflow.protocol import params
from sklearn.cluster import AgglomerativeClustering, KMeans
import time
import os
from sh_alignment.tompy.transform import fft, ifft, fftshift, ifftshift
import pwem.emlib.metadata as md
from continuousflex.protocols.utilities.spider_files3 import save_volume, open_volume
import xmipp3
from pwem.objects import Volume
import numpy as np
from sklearn import decomposition
from joblib import dump, load
from pwem.utils import runProgram


class FlexProtSubtomoClassify(ProtAnalysis3D):
    """ Protocol applying post alignment classification on subtomograms. """
    _label = 'classify subtomograms'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('SubtomoSource', EnumParam, default=0,
                      label='Source of Subtomograms',
                      choices=['After subtomogram synthesis', 'After subtomogram averaging'],
                      help='Choose the source of the subtomograms that you want to classify')
        form.addParam('ProtSynthesize', params.PointerParam, pointerClass='FlexProtSynthesizeSubtomo',
                      condition='SubtomoSource == 0',
                      label="Subtomogram synthesis",
                      help='All PDBs should have the same size')
        form.addParam('StA',params.PointerParam,
                      condition='SubtomoSource == 1',
                      pointerClass='FlexProtSubtomogramAveraging',
                      label="StA protocol",
                      help='Choose a subtomogram averaging previous run')
        form.addParam('classifyTechnique', EnumParam, default=0,
                      label='Classification techinque',
                      choices=['Hierarchical clustering', 'Dimentionality reduction then Clustering'],
                      help='Choose a classification techinque')
        form.addParam('ClusteringLinkage', EnumParam, default=0,
                      label='Linkage',
                      condition='classifyTechnique == 0',
                      choices=['ward'])
        form.addParam('dimredMethod', EnumParam, default=0,
                      condition='classifyTechnique == 1',
                      choices=['Scikit-Learn PCA'],
                      label='Dimensionality reduction method',
                      help='This method will be used to reduce the dimensions of the covariance matrix')
        form.addParam('applyMask', params.BooleanParam, label='Use a mask?', default=False,
                       help='This mask will be applied on the aligned particles before finding the cross correlation'
                            ', a proper mask is a mask derived from the subtomogram average (tight), that can be '
                            'applied on it without cropping any part.')
        form.addParam('Mask', params.PointerParam,
                       condition='applyMask',
                       pointerClass='Volume', allowsNull=True,
                       label="Select mask")
        form.addParam('reducedDim', IntParam, default=2,
                      condition='classifyTechnique == 1',
                      label='Reduced dimension')
        form.addParam('numOfClasses', IntParam, default=2,
                      label='Number of classes')
        # form.addParallelSection(threads=0, mpi=8)

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        mdSubtomo = self.getSubtomoMetaData()
        self._insertFunctionStep('subtomo_wedge_align',mdSubtomo)
        self._insertFunctionStep('find_covariance_matrix')
        if self.classifyTechnique.get() == 0:
            self._insertFunctionStep('performHierarchicalClustering')
        else:
            self._insertFunctionStep('performKmeansClustering')
        self._insertFunctionStep('findTotalAverage')
        self._insertFunctionStep('createOutputStep')


    # --------------------------- STEPS functions --------------------------------------------
    def subtomo_wedge_align(self,mdSubtomo):
        # we align the subtomograms but we align also a missing wedge mask for each one
        tilt = self.getTiltRange()
        # Creating a missing wedge mask:
        start_ang = tilt[0]
        end_ang = tilt[1]
        leg = self.getVolumeSize()
        size = (leg, leg, leg)
        MW_mask = np.ones(size)
        x, z = np.mgrid[0.:size[0], 0.:size[2]]
        x -= size[0] / 2
        ind = np.where(x)
        z -= size[2] / 2
        angles = np.zeros(z.shape)
        angles[ind] = np.arctan(z[ind] / x[ind]) * 180 / np.pi
        angles = np.reshape(angles, (size[0], 1, size[2]))
        angles = np.repeat(angles, size[1], axis=1)
        MW_mask[angles > -start_ang] = 0
        MW_mask[angles < -end_ang] = 0
        MW_mask[size[0] // 2, :, :] = 0
        MW_mask[size[0] // 2, :, size[2] // 2] = 1
        fnmask = self._getExtraPath('missing_wedge.spi')
        save_volume(np.float32(MW_mask), fnmask)
        args = ' -i ' + fnmask + ' --rotate_volume euler 0 90 0'
        runProgram('xmipp_transform_geometry',args)
        # Now aligning each subtomogram and its missing wedge version
        mw_path = self._getExtraPath('mw_masks/')
        subtom_aligned_path = self._getExtraPath('aligned_subtomograms/')
        makePath(mw_path)
        makePath(subtom_aligned_path)
        subtomogramMD = md.MetaData(mdSubtomo)
        subtomogaligneMD = md.MetaData()
        mwalignedMD = md.MetaData()
        for i in subtomogramMD:
            fnsubtomo = subtomogramMD.getValue(md.MDL_IMAGE, i)
            bnsubtomo = os.path.basename(fnsubtomo)
            bnwedge = removeBaseExt(bnsubtomo)+'_wedge.spi'
            fnalignedsubtomo = self._getExtraPath('aligned_subtomograms/'+bnsubtomo)
            fnalignedmask = self._getExtraPath('mw_masks/'+bnwedge)
            # print(fnalignedsubtomo)
            # print(fnalignedmask)
            rot = str(subtomogramMD.getValue(md.MDL_ANGLE_ROT, i))
            tilt = str(subtomogramMD.getValue(md.MDL_ANGLE_TILT, i))
            psi = str(subtomogramMD.getValue(md.MDL_ANGLE_PSI, i))
            shiftx = str(subtomogramMD.getValue(md.MDL_SHIFT_X, i))
            shifty = str(subtomogramMD.getValue(md.MDL_SHIFT_Y, i))
            shiftz = str(subtomogramMD.getValue(md.MDL_SHIFT_Z, i))
            # align the subtomogram

            if self.getAngleY() == 90:
                args = '-i ' + fnsubtomo + ' -o ' + fnalignedsubtomo + ' --rotate_volume euler 0 90 0'
                runProgram('xmipp_transform_geometry', args)
                params = '-i ' + fnalignedsubtomo + ' -o ' + fnalignedsubtomo + ' '

            else:
                params = '-i ' + fnsubtomo + ' -o ' + fnalignedsubtomo + ' '
                params += ' --inverse '

            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '

            runProgram('xmipp_transform_geometry', params)

            # align the mask (no shift should be applied only angles)
            if self.getAngleY() == 90:
                args = '-i ' + fnmask + ' -o ' + fnalignedmask + ' --rotate_volume euler 0 90 0'
                runProgram('xmipp_transform_geometry', args)
                params = '-i ' + fnalignedmask + ' -o ' + fnalignedmask + ' '
            else:
                params = '-i ' + fnmask + ' -o ' + fnalignedmask + ' '
                params += ' --inverse '

            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '

            runProgram('xmipp_transform_geometry', params)

            if (self.applyMask):
                maskfn = self.Mask.get().getFileName()
                params = '-i ' + fnalignedsubtomo + ' -o ' + fnalignedsubtomo + ' --mult ' + maskfn
                runProgram('xmipp_image_operate', params)

            subtomogaligneMD.setValue(md.MDL_IMAGE, fnalignedsubtomo, subtomogaligneMD.addObject())
            mwalignedMD.setValue(md.MDL_IMAGE, fnalignedmask, mwalignedMD.addObject())
        subtomogaligneMD.write(self._getExtraPath('aligned_subtomograms.xmd'))
        mwalignedMD.write(self._getExtraPath('aligned_masks.xmd'))




    def find_covariance_matrix(self):
        # if the covariance matrix exists, we should not re-evaluate it (it takes long time)
        fn_covarmat = self._getExtraPath('covar_mat.pkl')
        if os.path.exists(fn_covarmat):
            pass
        subtomogaligneMD= md.MetaData(self._getExtraPath('aligned_subtomograms.xmd'))
        mwalignedMD= md.MetaData(self._getExtraPath('aligned_masks.xmd'))
        N = subtomogaligneMD.size()
        X = np.zeros([N, N])
        print(N)
        A = time.time()
        for i in subtomogaligneMD:
            if i == 2:
                B = time.time()-A
                print('estimated time to finish is ', B*N/2)
            name_i = subtomogaligneMD.getValue(md.MDL_IMAGE, i)
            wedge_i = mwalignedMD.getValue(md.MDL_IMAGE, i)
            Vi = open_volume(name_i)
            Wi = open_volume(wedge_i)
            print('line is', str(i),' out of ',str(N))
            for j in range(i,N+1):
                name_j = subtomogaligneMD.getValue(md.MDL_IMAGE, j)
                wedge_j = mwalignedMD.getValue(md.MDL_IMAGE, j)
                Vj = open_volume(name_j)
                Wj = open_volume(wedge_j)
                Omega = Wi * Wj
                FVi = fft(Vi)
                FVi = fftshift(FVi) * Omega
                FVi = ifftshift(FVi)
                Vi_p = ifft(FVi)
                FVj = fft(Vj)
                FVj = fftshift(FVj) * Omega
                FVj = ifftshift(FVj)
                Vj_p = ifft(FVj)
                Vi_p = np.array(Vi_p, dtype=np.float32)
                Vj_p = np.array(Vj_p, dtype=np.float32)
                X[i-1, j-1] = self.cc(Vi_p, Vj_p)
                X[j-1, i-1] = self.cc(Vj_p, Vi_p)
        # save the covariance matrix:
        dump(X, fn_covarmat)

    def performHierarchicalClustering(self):
        fn_covarmat = self._getExtraPath('covar_mat.pkl')
        data = load(fn_covarmat)
        # 1 - CCCij (to keep with the literature)
        data = np.ones_like(data) - data
        clustering = AgglomerativeClustering(n_clusters=self.numOfClasses.get(), linkage='ward')
        clustering_class = clustering.fit_predict(data)
        labels = clustering.labels_
        # for l in np.unique(label):
        #     print(len(data[label == l]))
        dump(labels, filename=self._getExtraPath('hierarchical_clustering_labels.pkl'))
        print(clustering_class)
        # creating a metadate of subtomograms for each class and an average
        N = self.getVolumeSize()
        Averages = np.zeros([N, N, N, self.numOfClasses.get()])
        subtomogaligneMD = md.MetaData(self._getExtraPath('aligned_subtomograms.xmd'))
        # creating a metadata for each class
        classesMD = [md.MetaData() for i in range(self.numOfClasses.get())]
        for i in subtomogaligneMD:
            name = subtomogaligneMD.getValue(md.MDL_IMAGE, i)
            # the metadata index start from 1, we should subtract one to match the matrix
            label = labels[i-1]
            vol = open_volume(name)
            Averages[:, :, :, label] += vol
            classesMD[label].setValue(md.MDL_IMAGE, name, classesMD[label].addObject())
        for i in range(self.numOfClasses.get()):
            name = self._getExtraPath('class_'+str(i).zfill(2)+'.xmd')
            classesMD[i].write(name)
        # creating a metadata for all class averages:
        md_averages = md.MetaData()
        makePath(self._getExtraPath('class_averages/'))
        for j in range(self.numOfClasses.get()):
            num = len(labels[labels == j])
            print('Number of subtomograms in class #', j, 'is ', num)
            name = self._getExtraPath('class_averages/') + 'cluster' + str(j).zfill(2) + '.spi'
            ave = np.array(Averages[:, :, :, j] / num, dtype=np.float32)
            save_volume(ave, name)
            md_averages.setValue(md.MDL_IMAGE,name,md_averages.addObject())
        md_averages.write(self._getExtraPath('averages.xmd'))


    def performKmeansClustering(self):
        X = load(self._getExtraPath('covar_mat.pkl'))
        pca = decomposition.PCA(n_components=self.reducedDim.get())
        pca.fit(X)
        data = pca.transform(X)
        pca_pickled = self._getExtraPath('pca_pickled.pkl')
        np.savetxt(self._getExtraPath('dimred_mat.txt'),data)
        dump(pca,pca_pickled)
        # clustering now
        clustering = KMeans(n_clusters=self.numOfClasses.get()).fit(data)
        clustering_class = clustering.fit_predict(data)
        labels = clustering.labels_
        # for l in np.unique(label):
        #     print(len(data[label == l]))
        dump(clustering, filename=self._getExtraPath('kmeans_algo.pkl'))
        dump(labels, filename=self._getExtraPath('kmeans_clustering_labels.pkl'))
        print(clustering_class)
        # creating a metadate of subtomograms for each class and an average
        N = self.getVolumeSize()
        Averages = np.zeros([N, N, N, self.numOfClasses.get()])
        subtomogaligneMD = md.MetaData(self._getExtraPath('aligned_subtomograms.xmd'))
        # creating a metadata for each class
        classesMD = [md.MetaData() for i in range(self.numOfClasses.get())]
        for i in subtomogaligneMD:
            name = subtomogaligneMD.getValue(md.MDL_IMAGE, i)
            # the metadata index start from 1, we should subtract one to match the matrix
            label = labels[i - 1]
            vol = open_volume(name)
            Averages[:, :, :, label] += vol
            classesMD[label].setValue(md.MDL_IMAGE, name, classesMD[label].addObject())
        for i in range(self.numOfClasses.get()):
            name = self._getExtraPath('class_' + str(i).zfill(2) + '.xmd')
            classesMD[i].write(name)
        # creating a metadata for all class averages:
        md_averages = md.MetaData()
        makePath(self._getExtraPath('class_averages/'))
        for j in range(self.numOfClasses.get()):
            num = len(labels[labels == j])
            print('Number of subtomograms in class #', j, 'is ', num)
            name = self._getExtraPath('class_averages/') + 'cluster' + str(j).zfill(2) + '.spi'
            ave = np.array(Averages[:, :, :, j] / num, dtype=np.float32)
            save_volume(ave, name)
            md_averages.setValue(md.MDL_IMAGE, name, md_averages.addObject())
        md_averages.write(self._getExtraPath('averages.xmd'))
        pass

    def findTotalAverage(self):
        N = self.getVolumeSize()
        Average = np.zeros([N, N, N])
        subtomogaligneMD = md.MetaData(self._getExtraPath('aligned_subtomograms.xmd'))
        count = 0
        for i in subtomogaligneMD:
            name = subtomogaligneMD.getValue(md.MDL_IMAGE, i)
            vol = open_volume(name)
            Average[:, :, :] += vol
            count+=1
        name = self._getExtraPath('global_average.spi')
        save_volume(np.array(Average[:, :, :] / count, dtype=np.float32), name)
        pass

    def createOutputStep(self):
        out_mdfn = self._getExtraPath('averages.xmd')
        partSet = self._createSetOfVolumes('Averages')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.getSamplingRate())
        outvolume = Volume()
        outvolume.setSamplingRate(self.getSamplingRate())
        outvolume.setFileName(self._getExtraPath('global_average.spi'))
        self._defineOutputs(ClassAvarages=partSet, GlobalAverage=outvolume)
        pass

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2022continuousflex']

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

    def getSubtomoMetaData(self):
        if self.SubtomoSource.get()==0:
            return self.ProtSynthesize.get()._getExtraPath('GroundTruth.xmd')
        if self.SubtomoSource.get()==1:
            return self.StA.get()._getExtraPath('final_md.xmd')

    def getTiltRange(self):
        if self.SubtomoSource.get()==0:
            return [self.ProtSynthesize.get().tiltLow.get(), self.ProtSynthesize.get().tiltHigh.get()]
        if self.SubtomoSource.get()==1:
            return [self.StA.get().tiltLow.get(), self.StA.get().tiltHigh.get()]

    def getAngleY(self):
        if self.SubtomoSource.get() == 0:
            return 0
        if self.SubtomoSource.get() == 1:
            temp = md.MetaData(self.StA.get()._getExtraPath('final_md.xmd'))
            return temp.getValue(md.MDL_ANGLE_Y, 1)


    def getSamplingRate(self):
        if self.SubtomoSource.get()==0:
            return self.ProtSynthesize.get().samplingRate.get()
        if self.SubtomoSource.get()==1:
            return self.StA.get().inputVolumes.get().getSamplingRate()

    def getVolumeSize(self):
        if self.SubtomoSource.get() == 0:
            return self.ProtSynthesize.get().volumeSize.get()
        if self.SubtomoSource.get() == 1:
            return self.StA.get().SubtomogramAverage.getDim()[0]


    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def getDeformationFile(self):
        return self._getExtraPath('pdbs_mat.txt')

    def normalize(self,v):
        """Normalize a volume.
        @param v: input volume.
        @return: Normalized volume.
        """
        m = np.mean(v)
        v = v - m
        s = np.std(v)
        v = v / s
        return v

    def cc(self, v1, v2):
        """Compute the Normalized Cross Correlation between the two volumes.
        @param v1: volume 1.
        @param v2: volume 2.
        @return: NCC.
        """
        vv1 = self.normalize(v1)
        vv2 = self.normalize(v2)
        score = np.sum(vv1 * vv2) / vv1.size
        return score
