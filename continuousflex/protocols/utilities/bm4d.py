import continuousflex
import os
import tkinter.messagebox as tk


def bm4d(path_vol_in, path_vol_out, distribution, sigma, profile, do_wiener):
    # Function to run the bm4d matlab wrapper
    wrapper_path = continuousflex.__path__[0] + '/protocols/utilities/'
    if os.getenv('MATLAB_HOME') is None:
        tk.showerror('Error', 'MATLAB_HOME is not set in your path, it should be set to use this method. '
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
    os.system(command)
    pass
    
# bm4d('noisy_volume.vol','denoised_volume.spi','Gauss',0,'mp',0)
