% SYNTAX: figure; volxyz(data, range)
% Plots orthoslices of 3D volume/

% by E. Moebel

function [] = volxyz(data, range)

if nargin<2
    mu = mean(data(:));
    sig = std(data(:));
    k = 5;
    range = [mu-k*sig mu+k*sig];
end

sliceNb = round(size(data)./2); % was floor instead of round

slice1 = data(:,:,sliceNb(3)); % <= faster
slice2 = data(:,sliceNb(2),:);
slice3 = data(sliceNb(1),:,:);

slice2 = permute(slice2, [1 3 2]);
slice3 = permute(slice3, [3 2 1]);

dimImg = [size(slice1,1)+size(slice3,1)+1, size(slice1,2)+size(slice2,2)+1];
img = zeros(dimImg);

img(1:size(slice1,1), 1:size(slice1,2)) = slice1;
img(1:size(slice1,1), size(slice1,2)+2:end) = slice2;
img(size(slice1,1)+2:end, 1:size(slice1,2)) = slice3;

imshow(img, range);

hold on; % plot lines to separate different views
line([size(slice1,2)+1 size(slice1,2)+1], [1 size(img,1)], 'Color', 'white', 'LineWidth', 2);
line([size(img,2) 1], [size(slice1,1)+1 size(slice1,1)+1], 'Color', 'white', 'LineWidth', 2);
rectangle('Position', [size(slice1,2)+1 size(slice1,1)+1 size(slice1,2) size(slice1,2)], 'FaceColor', [1 1 1], 'EdgeColor', [1 1 1]);

