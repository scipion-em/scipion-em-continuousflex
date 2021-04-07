% SYNTAX: figure; plot_spectrum_3D(data, range);
% Inputs:
%    data : 3D data in spatial domain
%    range: display range

% By E. Moebel

function [] = plot_spectrum_3D(data, range)

if nargin<2
    range = [];
end

volxyzColor(fftshift(log(abs(fftn(data)))), range);