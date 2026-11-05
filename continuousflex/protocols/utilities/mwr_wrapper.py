# By Mohamad Harastani
import continuousflex
import os
import subprocess

# TODO: deprecate (MATLAB-only, unmaintained) together with the protocol that calls it


def mwr(path_vol_in, path_wedge, path_vol_out, sigma_noise, T, Tb, beta, mask_shifted):
    # Function to run the missing wedge restoration matlab wrapper
    wrapper_path = continuousflex.__path__[0] + '/protocols/utilities/'
    if os.getenv('MATLAB_HOME') is None:
        raise RuntimeError('MATLAB_HOME is not set in your path, it should be set to use this method. '
                           'We assume that matlab executable is at $MATLAB_HOME/bin/matlab')
    matlab = os.getenv('MATLAB_HOME') + '/bin/matlab'
    command = matlab
    command += ' -nodisplay -nosplash -nodesktop -r '
    command += '"addpath(genpath('
    command += "'" + wrapper_path + "'));mwr_wrapper("
    command += "'" + path_vol_in + "', '"
    command += path_wedge + "', '"
    command += path_vol_out + "', "
    command += str(sigma_noise) + ", "
    command += str(T) + ", "
    command += str(Tb) + ", "
    command += str(beta) + ", "
    if mask_shifted:
        command += 'true'
    command += ');exit;"'
    # print(command)
    subprocess.check_call(command, shell=True)
    if not os.path.isfile(path_vol_out):
        raise RuntimeError('MATLAB did not write ' + path_vol_out)
