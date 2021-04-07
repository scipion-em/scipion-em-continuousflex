% This scripts applies the missing wedge restoration (MWR) to synthetic
% data and plots the results.
% by E. Moebel

%close all;
clear all;

addpath(genpath('utils/'));

% Load data:
Vin = load('data/proteasome_data.mat');
Vin = Vin.data;
% Load ground truth:
Vgt = load('data/proteasome_gt.mat');
Vgt = Vgt.gt;
% Load wedge:
wedge = load('data/proteasome_wedge.mat');
wedge = wedge.wedge;
wedge_shifted = fftshift(wedge);
writeSPIDERfile('wedge_shifted.spi', wedge_shifted);

% Set parameters:
sigma_noise = 0.2;
plotFlag    = 1; % set to 1 to observe processing in real time

% Launch processing:
Vout = mwr(Vin, sigma_noise, wedge, plotFlag);

% Plot result:
mplot = 2;
nplot = 3;
plotrange = [0 1];
plotrangeS = [-12 12];
figure;
subplot(mplot,nplot,1); volxyz(1-Vgt, plotrange); freezeColors; title('Ground truth');
subplot(mplot,nplot,2); volxyz(1-Vin, plotrange); freezeColors; title('Noisy data');
subplot(mplot,nplot,3); volxyz(1-Vout, plotrange); freezeColors; title('Processed data');
subplot(mplot,nplot,4); plot_spectrum_3D(Vgt , plotrangeS); freezeColors; title('Fourier');
subplot(mplot,nplot,5); plot_spectrum_3D(Vin , plotrangeS); freezeColors; title('Fourier');
subplot(mplot,nplot,6); plot_spectrum_3D(Vout, plotrangeS); freezeColors; title('Fourier');

% Plot performance measures:
figure;
subplot(121);
    bar([psnr(Vin, Vgt, 1) psnr(Vout, Vgt, 1)]);
    grid on;
    set(gca, 'xticklabel', {'before restoration', 'after restoration'});
    ylabel('PSNR');
subplot(122);
    bar([ccc(Vin, Vgt, 1-wedge) ccc(Vout, Vgt, 1-wedge)]);
    grid on;
    set(gca, 'xticklabel', {'before restoration', 'after restoration'});
    ylabel('CCC');