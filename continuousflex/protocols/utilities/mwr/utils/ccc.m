% SYNTAX: ccc = ccc(A,B,wedge)
% Computes the correlation coefficient in Fourier domain, only using
% Fourier coefficients defined by the binary mask 'wedge'.

% by E. Moebel

function ccc = ccc(A,B,wedge)

A = (A-mean(A(:)))./std(A(:));
B = (B-mean(B(:)))./std(B(:));

wedge = logical(wedge);

Fa = fftn(A);
Fb = fftn(B);
num = sum(Fa(wedge).*conj(Fb(wedge)));
den = sqrt( sum(abs(Fa(wedge)).^2) * sum(abs(Fb(wedge)).^2) );
ccc = real( num/den );