% SYNTAX: mwr_wrapper(path_vol_in, path_wedge, path_vol_out, sigma_noise, T, Tb, beta, mask_shifted)
% Arguments:
%     path_vol_in   : path for input data volume (spider format)
%     path_wedge    : binary mask (3D array) defining the support (in Fourier domain) of the missing
%                     wedge (0 for missing wedge, 1 else) (spider format)
%     path_vol_out  : path for the restored data volume (spider format)
%     sigma_noise   : estimated standart deviation of data noise
%                     defines the strength of the processing (high value gives smooth images)
%     T             : number of iterations (default: 300)
%     Tb            : length of the burn-in phase, i.e. first Tb samples are
%                     discarded (default: 100)
%     beta          : scale parameter, affects the acceptance rate (default: 0.00004)
%     mask_shifted  : if the missing wedge mask is visually correct applied
%                     shifting the 0 frequency to center when generated, then this should be true,
%                     if it is maintaind as missing fftn region then false (default is false)


% A wrapper for the implementation of the method described in
% Emmanuel Moebel, Charles Kervrann,
% A Monte Carlo framework for missing wedge restoration and noise removal in cryo-electron tomography,
% Journal of Structural Biology: X,
% Volume 4,
% 2020,
% 100013,
% ISSN 2590-1524,
% https://doi.org/10.1016/j.yjsbx.2019.100013.
% (https://www.sciencedirect.com/science/article/pii/S259015241930011X)

% by Mohamad Harastani (mohamad.harastani@igbmc.fr)


function mwr_wrapper(path_vol_in, path_wedge, path_vol_out, sigma_noise, T, Tb, beta, mask_shifted)
if nargin<8; mask_shifted = false ; end
if nargin<7; beta = 0.00004       ; end
if nargin<6; Tb   = 100           ; end
if nargin<5; T    = 300           ; end
if nargin<4; sigma_noise = 0.2    ; end

addpath(genpath('spider_matlab/'));
addpath(genpath('mwr/'));

disp(strcat('path for input volume:             ', path_vol_in))
disp(strcat('path for binary mask:              ', path_wedge))
disp(strcat('path for the restored data volume: ', path_vol_out))
disp(strcat('standard deviation of data noise:  ', string(sigma_noise)))
disp(strcat('number of iterations:              ', string(T)))
disp(strcat('length of the burn-in phase:       ', string(Tb)))
disp(strcat('scale parameter:                   ', string(beta)))
disp(strcat('mask_shifted flag:                 ', string(mask_shifted)))

Vin = readSPIDERfile(path_vol_in);
wedge = readSPIDERfile(path_wedge);

if(mask_shifted) % case the generated missing wedge mask is shifted
    wedge = ifftshift(wedge);
end

Vout = mwr(Vin, sigma_noise, wedge, 0, T, Tb, beta);

writeSPIDERfile(path_vol_out,Vout);

end