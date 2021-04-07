% SYNTAX: p = perdecomp3D(u)
% Computes periodic element of input (the fft of periodic element has no
% cross artefact)
% Input : u (3D array)
% Ouput : p (its periodic element)

% This code is a 3D adaptation of the implementation found in: http://www.mi.parisdescartes.fr/~moisan/p+s
% by E. Moebel, the 16/09/16

function p = perdecomp3D(u)

dim = size(u);

% Compute boundary image:
v = zeros(dim);
v(1  ,:,:) = u(1  ,:,:) - u(end,:,:);
v(end,:,:) = u(end,:,:) - u(1  ,:,:);
v(:,1  ,:) = v(:,1  ,:) + u(:,1  ,:) - u(:,end,:);
v(:,end,:) = v(:,end,:) + u(:,end,:) - u(:,1  ,:);
v(:,:,1  ) = v(:,:,1  ) + u(:,:,1  ) - u(:,:,end);
v(:,:,end) = v(:,:,end) + u(:,:,end) - u(:,:,1  );

% Compute periodic element:
idx1 = 1:dim(1); % define indexes for each dim
idx2 = 1:dim(2);
idx3 = 1:dim(3);

idx1 = idx1'; % orient the indexes in space
idx3 = permute(idx3, [1 3 2]);

f1 = repmat(cos(2.*pi.*(idx1-1)./dim(1)), 1     , dim(2), dim(3));
f2 = repmat(cos(2.*pi.*(idx2-1)./dim(2)), dim(1), 1     , dim(3));
f3 = repmat(cos(2.*pi.*(idx3-1)./dim(3)), dim(1), dim(2), 1     );

f1(1,1,1) = 0; % avoid division by 0 in the line below
s = real(ifftn( fftn(v)./(6-2*f1-2*f2-2*f3) ));
p = u-s;
