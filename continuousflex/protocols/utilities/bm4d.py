# By Mohamad Harastani

import continuousflex
import os
import subprocess

# TODO: deprecate (MATLAB-only, unmaintained) together with the protocol that calls it


def bm4d(path_vol_in, path_vol_out, distribution, sigma, profile, do_wiener):
    # Function to run the bm4d matlab wrapper
    wrapper_path = continuousflex.__path__[0] + '/protocols/utilities/'
    if os.getenv('MATLAB_HOME') is None:
        raise RuntimeError('MATLAB_HOME is not set in your path, it should be set to use this method. '
                           'We assume that matlab executable is at $MATLAB_HOME/bin/matlab')
    matlab = os.getenv('MATLAB_HOME') + '/bin/matlab'
    command = matlab
    command += ' -nodisplay -nosplash -nodesktop -r '
    command += '"addpath(genpath('
    command += "'" + wrapper_path + "'));bm4d_wrapper("
    command += "'" + path_vol_in + "', '"
    command += path_vol_out + "', '"
    command += distribution + "', "
    command += str(sigma) + ", '"
    command += profile + "', "
    command += str(do_wiener)
    command += ');exit;"'
    # print(command)
    subprocess.check_call(command, shell=True)
    if not os.path.isfile(path_vol_out):
        raise RuntimeError('MATLAB did not write ' + path_vol_out)
    
# bm4d('noisy_volume.vol','denoised_volume.spi','Gauss',0,'mp',0)
