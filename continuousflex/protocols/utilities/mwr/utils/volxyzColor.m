% SYNTAX: figure; volxyzColor(data, range)
% Plots orthoslices of volume, in color.

% by E. Moebel

function [] = volxyzColor(data, range)

if nargin<2
    range = [];
end

sliceNb = round(size(data)./2)+1;
data2 = permute(data, [1 3 2]);
data3 = permute(data, [3 2 1]);

slice1 = data(:,:,sliceNb(3));
slice2 = data2(:,:,sliceNb(2));
slice3 = data3(:,:,sliceNb(1));

dimImg = [size(slice1,1)+size(slice3,1)+1, size(slice1,2)+size(slice2,2)+1];
img = zeros(dimImg);

img(1:size(slice1,1), 1:size(slice1,2)) = slice1;
img(1:size(slice1,1), size(slice1,2)+2:end) = slice2;
img(size(slice1,1)+2:end, 1:size(slice1,2)) = slice3;

imshow(img, range, 'colormap', jet(255));
% imshow(img, range, 'colormap', hot);

hold on; % plot lines to separate different views
line([size(slice1,2)+1 size(slice1,2)+1], [1 size(img,1)], 'Color', 'white', 'LineWidth', 2);
line([size(img,2) 1], [size(slice1,1)+1 size(slice1,1)+1], 'Color', 'white', 'LineWidth', 2);
rectangle('Position', [size(slice1,2)+1 size(slice1,1)+1 size(slice1,2) size(slice1,2)], 'FaceColor', [1 1 1], 'EdgeColor', [1 1 1]);
