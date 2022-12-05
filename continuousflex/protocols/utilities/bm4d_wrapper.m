% SYNTAX: bm4d_wrapper(path_vol_in, path_vol_out, distribution, sigma, profile, do_wiener)
% Arguments:
%     path_vol_in          : path for input data volume (spider format)
%     path_vol_out         : path for the restored data volume (spider format)
%     distribution  (char) : 'Gauss' --> z has Gaussian distribution
%                          : 'Rice'  --> z has Rician distribution
%     sigma  (double)      : Noise standard deviation, if unknown 
%                            set it to 0 to enable noise estimation
%                             (default is 0)
%     profile (char)       : 'lc' --> low complexity profile 
%                          : 'np' --> normal profile 
%                          : 'mp' --> modified profile
%                             (default is 'mp')
%     do_wiener (logical)  : Perform collaborative Wiener filtering
%                             (default is 1)


% A wrapper for the implementation of the method described in
%      M. Maggioni, V. Katkovnik, K. Egiazarian, A. Foi, "A Nonlocal
%      Transform-Domain Filter for Volumetric Data Denoising and Reconstruction", IEEE Trans. Image Process., vol. 22, no. 1,
%      M. Maggioni, A. Foi, "Nonlocal Transform-Domain Denoising of Volumetric Data With Groupwise Adaptive Variance Estimation", 
%      Proc. SPIE Electronic Imaging 2012, San Francisco, CA, USA, Jan. 2012.


% by Mohamad Harastani (mohamad.harastani@igbmc.fr)


function bm4d_wrapper(path_vol_in, path_vol_out, distribution, sigma, profile, do_wiener)
if nargin<6; do_wiener = 1    ; end
if nargin<5; profile   = 'mp' ; end
if nargin<4; sigma     = 0    ; end

addpath(genpath('spider_matlab/'));
addpath(genpath('mwr/'));

disp(strcat('path for input volume:             ', path_vol_in))
disp(strcat('path for the denoised volume:      ', path_vol_out))
disp(strcat('noise distribution:                ', distribution))
disp(strcat('standard deviation of data noise:  ', string(sigma)))
disp(strcat('nosie profile:                     ', profile))
disp(strcat('Wiener filter option:              ', string(do_wiener)))

Vin = readSPIDERfile(path_vol_in);

[Vout, sigma_est] = bm4d(Vin, distribution, sigma, profile, do_wiener, 0);
% disp(strcat('estimated sigma noise  ', string(sigma_est)))

writeSPIDERfile(path_vol_out,Vout);
% writeSPIDERfile('sigma.spi',sigma_est);

end
