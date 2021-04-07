clear all
close all
clc
% read a SPIDER volume
vol = readSPIDERfile('001.spi');

k=32; % the slice
figure
h = slice(double(vol),k,k,k);
%set(h,'FaceColor','interp','EdgeColor','none')
%colormap gray

A = squeeze(mat2gray(vol(:,32,:)));
figure
imshow(A);
figure 
imshow3D(squeeze(mat2gray(vol)));