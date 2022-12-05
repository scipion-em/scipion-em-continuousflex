# Author: Ilyes Hamitouche

from torchvision import transforms
from torch.utils.data import DataLoader
from processing_dh.data import cryodata
from processing_dh.utils import quater2euler, reverse_min_max
from processing_dh.models import deephemnma
import numpy as np
import torch
from pathlib import Path
import sys
import pwem.emlib.metadata as md
def norm(imgs_path, weights_path, flag, mode, batch_size):
    dataset = cryodata(imgs_path, weights_path, flag=flag, mode=mode, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=batch_size)
    sum_, squared_sum_, num_batches = 0, 0, 0
    for img, image_name in train_loader:
        sum_ += torch.mean(img, dim=[0, 2, 3])
        squared_sum_ += torch.mean(img**2, dim=[0, 2, 3])
        num_batches += 1
    mean = sum_/num_batches
    std = (squared_sum_/num_batches - mean**2)**0.5
    return mean, std
def infer(imgs_path, weights_path, output_path, num_modes, batch_size=2, flag=0, device=0, mode='inference'):
    FLAG = ''
    if flag==0:
        FLAG = 'nma'
    elif flag==1:
        FLAG = 'ang'
    elif flag==2:
        FLAG = 'shf'
    else:
        FLAG = 'all'
    DEVICE = ''
    if device==0:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    mean, std = norm(imgs_path, weights_path, flag=FLAG, mode=mode, batch_size=batch_size)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))])
    dataset = cryodata(imgs_path, weights_path, flag=FLAG, mode=mode, transform=transform)

    dataset_size = len(dataset)
    print('the train set size is: {} images'.format(dataset_size))

    data_loader = DataLoader(dataset, batch_size=batch_size)

    if FLAG=='nma':
        model = deephemnma(3).to(DEVICE)
        predictions = np.zeros((dataset_size, num_modes), dtype='float32')
    elif FLAG=='ang':
        model = deephemnma(4).to(DEVICE)
        predictions = np.zeros((dataset_size, 4), dtype='float32')
    elif FLAG=='shf':
        model = deephemnma(2).to(DEVICE)
        predictions = np.zeros((dataset_size, 2), dtype='float32')
    elif FLAG=='all':
        model = deephemnma(6+num_modes).to(DEVICE)
        predictions = np.zeros((dataset_size, 6+num_modes), dtype='float32')

    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():
        i = 0
        for img, image_name in data_loader:
            pred_params = model(img.to(DEVICE), mode)
            predictions[i * batch_size:(i + 1) * batch_size, :] = pred_params.cpu()
            i+=1


    if FLAG=='nma':
        min_max_nma = np.loadtxt(str(Path(weights_path).parent) + '/min_max_nma.txt')
        nma = reverse_min_max(predictions, min_max_nma[0], min_max_nma[1])
    elif FLAG=='ang':
        angles = predictions
        euler_angles = []
        for i in range(len(angles)):
            euler_angles.append(quater2euler(angles))
        euler_angles = np.array(euler_angles)
    elif FLAG=='shf':
        min_max_shf = np.loadtxt(str(Path(weights_path).parent) + '/min_max_shf.txt')
        shifts = reverse_min_max(predictions, min_max_shf[0], min_max_shf[1])
    elif FLAG=='all':
        min_max_nma = np.loadtxt(str(Path(weights_path).parent) + '/min_max_nma.txt')
        min_max_shf = np.loadtxt(str(Path(weights_path).parent) + '/min_max_shf.txt')
        nma = reverse_min_max(predictions[:,:num_modes], min_max_nma[0], min_max_nma[1])
        angles = predictions[:,num_modes:num_modes+4]
        shifts = reverse_min_max(predictions[:,num_modes+4:], min_max_shf[0], min_max_shf[1])
        euler_angles = []
        for i in range(len(angles)):
            euler_angles.append(quater2euler(angles[i]))
        euler_angles = np.array(euler_angles)
        mdImgs = md.MetaData(imgs_path)
        imgPath = []
        for objId in mdImgs:
            imgPath.append(mdImgs.getValue(md.MDL_IMAGE, objId))
        with open(output_path+'/images.xmd', 'w') as f:
            f.write('# XMIPP_STAR_1 * \n # \ndata_noname\nloop_\n _image\n _enabled\n _angleRot\n _angleTilt\n _anglePsi\n _shiftX\n _shiftY\n _nmaDisplacements\n _cost\n _itemId\n')
            for i in range(euler_angles.shape[0]):
                f.write(imgPath[i]+"                    1 {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} '{:>12.6} {:>12.6} {:>12.6}'   0.55  {}".format(
                    euler_angles[i, 0], euler_angles[i, 1], euler_angles[i, 2], shifts[i, 0],
                        shifts[i, 1], nma[i,0], nma[i,1], nma[i,2], i+1) + '\n')
if __name__ == '__main__':
    infer(sys.argv[1],
          sys.argv[2],
          sys.argv[3],
          int(sys.argv[4]),
          int(sys.argv[5]),
          int(sys.argv[6]),
          int(sys.argv[7]))
