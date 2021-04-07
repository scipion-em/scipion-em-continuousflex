% This scripts applies the missing wedge restoration (MWR) to experimental
% data and plots the results.
% by E. Moebel

close all;
clear all;

addpath(genpath('utils/'));

% Load data:
Vin = load('data/gold_particle_data.mat');
Vin = Vin.data;
% Load wedge:
wedge = load('data/gold_particle_wedge.mat');
wedge = wedge.wedge;

% Set parameters:
sigma_noise = 0.05;
plotFlag    = 0r; % set to 1 to observe processing in real time

% Launch processing:
Vout = mwr(Vin, sigma_noise, wedge, plotFlag);

% Plot result:
mplot = 2;
nplot = 2;
plotrange = [0.25 0.75];
plotrangeS = [-7 7];
figure;
subplot(mplot,nplot,1); volxyz(Vin, plotrange); freezeColors; title('Noisy data');
subplot(mplot,nplot,2); volxyz(Vout, plotrange); freezeColors; title('Processed data');
subplot(mplot,nplot,3); plot_spectrum_3D(Vin , plotrangeS); freezeColors; title('Fourier');
subplot(mplot,nplot,4); plot_spectrum_3D(Vout, plotrangeS); freezeColors; title('Fourier');