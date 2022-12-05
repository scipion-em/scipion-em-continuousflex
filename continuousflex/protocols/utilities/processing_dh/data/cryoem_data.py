import glob

import numpy as np
from torch.utils.data import Dataset
from ..utils import spi2array, eul2quat, min_max
import torch
import pwem.emlib.metadata as md
class cryodata(Dataset):

    def __init__(self, path, output_path, flag='nma', mode = 'train', transform=None):
        self.path = path
        self.flag = flag
        self.mode = mode
        self.transform = transform
        mdImgs = md.MetaData(self.path)
        if mode == 'train':
            rot = []
            tilt = []
            psi = []
            nma = []
            shift_x = []
            shift_y = []
            imgPath = []
            for objId in mdImgs:
                imgPath.append(mdImgs.getValue(md.MDL_IMAGE, objId))
                rot.append(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
                tilt.append(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
                psi.append(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
                shift_x.append(mdImgs.getValue(md.MDL_SHIFT_X, objId))
                shift_y.append(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
                nma.append(mdImgs.getValue(md.MDL_NMA, objId))
            self.images_Path = imgPath
            rot_ = torch.tensor(rot)
            tilt_ = torch.tensor(tilt)
            psi_ = torch.tensor(psi)
            shiftx = torch.tensor(shift_x)
            shifty = torch.tensor(shift_y)

            self.angles = torch.column_stack((rot_, tilt_, psi_))
            self.quaternions = torch.zeros((self.angles.shape[0], 4), dtype=torch.float32)
            for i in range(len(self.angles)):
                self.quaternions[i, :] = torch.tensor(eul2quat(self.angles, i))
            self.shifts, min_shf, max_shf = min_max(torch.column_stack((shiftx, shifty)))
            self.amplitudes, min_nma, max_nma = min_max(torch.tensor(nma, dtype=torch.float32))

            min_max_nma = torch.row_stack((min_nma, max_nma))
            min_max_shf = torch.row_stack((min_shf, max_shf))
            np.savetxt(output_path + '/min_max_nma.txt', min_max_nma.numpy())
            np.savetxt(output_path + '/min_max_shf.txt', min_max_shf.numpy())
        elif self.mode=='inference':
            imgPath = []
            for objId in mdImgs:
                imgPath.append(mdImgs.getValue(md.MDL_IMAGE, objId))
            self.images_Path = imgPath
        else:
            pass



    def __len__(self):
        return len(self.images_Path)

    def __getitem__(self, item):
        if self.mode == 'train':
            if self.flag == 'nma':
                amplitudes = self.amplitudes[item]
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    amplitudes = torch.tensor(amplitudes)
                return spi_array, amplitudes
            elif self.flag == 'ang':
                angles = self.quaternions[item]
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    angles = torch.tensor(angles) 
                return spi_array, angles, image_name
            elif self.flag == 'shf':
                shifts = self.shifts[item]    
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    shifts = torch.tensor(shifts)
                return spi_array, shifts
            elif self.flag=='all':
                amplitudes = self.amplitudes[item]
                angles = self.quaternions[item]
                shifts = self.shifts[item]
                image_name = self.images_Path[item]
                spi_array = spi2array(image_name)
                if self.transform:
                    spi_array = self.transform(spi_array)
                    amplitudes = torch.tensor(amplitudes)
                    angles = torch.tensor(angles)
                    shifts = torch.tensor(shifts)
                    params = torch.cat([amplitudes, angles, shifts])
                return spi_array, params
        elif self.mode == 'inference':
            image_name = self.images_Path[item]
            spi_array = spi2array(image_name)
            if self.transform:
                spi_array = self.transform(spi_array)
            return spi_array, image_name
