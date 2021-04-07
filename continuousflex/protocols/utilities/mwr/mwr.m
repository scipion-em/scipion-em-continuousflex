% SYNTAX: Vout = mwr(Vin, sigma_noise, wedge, plotFlag, T, Tb, beta)
% Inputs:
%     Vin         : input data volume
%     sigma_noise : estimated standart deviation of data noise
%                   defines the strength of the processing (high value gives smooth images)
%     wedge    : binary mask (3D array) defining the support (in Fourier domain) of the missing
%                wedge (0 for missing wedge, 1 else)
%     plotFlag : set to 1 to observe the processing in real time (default: 0)
%                /!\ slows down the processing
%     T    : number of iterations (default: 300)
%     Tb   : length of the burn-in phase, i.e. first Tb samples are
%            discarded (default: 100)
%     beta : scale parameter, affects the acceptance rate (default: 0.00004)
% Output:
%     Vout : the restored data volume

% Implementation of the method described in E. Moebel & C. Kervrann "A Monte Carlo framework 
% for noise removal and missing wedge restoration in cryo-electron tomography"

% by E. Moebel

function Vout = mwr(Vin, sigma_noise, wedge, plotFlag, T, Tb, beta)

if nargin<7; beta = 0.00004; end;
if nargin<6; Tb   = 100    ; end;
if nargin<5; T    = 300    ; end;
if nargin<4; plotFlag = 0  ; end;
    
% Initialize variables:
sigma_excite  = sigma_noise;
dim           = size(Vin);
y             = Vin;
y_spectrum    = fftn(y);
x_mmse        = zeros(dim);
buffer        = zeros(dim);
reject_hist   = zeros(1,T);

x_initial = fftn(y+sigma_excite*randn(dim)); % add noise
x_initial(wedge==1) = y_spectrum(wedge==1);  % spectrum constraint
x_initial = real(ifftn(x_initial));
x_current = bm4d(x_initial, 'Gauss', sigma_noise, 'lc', 0, 0);

% Parameters for plotFlag=1:
plotrange = [mean(y(:))-5*std(y(:)) mean(y(:))+5*std(y(:))];
% plotrangeS = dynamic(log(abs(fftn(y))));
% plotrangeS = [-7, 7];
tmp = log(abs(fftn(y)));
plotrangeS = [min(tmp(:)), max(tmp(:))];
mplot = 2;
nplot = 3;


normFactor = 1;
tic;
for t = 1:T
    % Add noise:
    z = x_current + sigma_excite*randn(dim);
    
    % Spectrum constraint:
    z_spectrum           = fftn(z);
    z_spectrum(wedge==1) = y_spectrum(wedge==1);
    z                    = real( ifftn(z_spectrum) );
    
    % Denoise:
    z = denoise(z, sigma_excite);
    z = perdecomp3D(z);
    
    % Compute energies:
    Ucurrent(t)  = compute_energy(y, x_current , wedge);
    Uproposed(t) = compute_energy(y, z         , wedge);

    deltaU = Uproposed(t) - Ucurrent(t);
    deltaP = exp(-deltaU/beta);
    
    % Accept/reject according to Metropolis:
    ak = rand();
    if deltaU<0
        x_current  = z;
    else
        if ak<deltaP
            x_current  = z;
        else
            x_current = x_current;
            reject_hist(t) = 1;
        end
    end
    
    % Aggregation:
    if t>Tb
        buffer  = buffer + x_current;
        x_mmse  = buffer./normFactor;
        normFactor = normFactor+1;
    end
    
    if plotFlag==1
    figure(1); 
    subplot(mplot,nplot,1); volxyz(y     , plotrange); freezeColors; title('y: input volume');
    subplot(mplot,nplot,2); volxyz(z     , plotrange); freezeColors; title(['z: generated sample (reject:', num2str(reject_hist(t)), ')']);
    subplot(mplot,nplot,3); volxyz(x_mmse, plotrange); freezeColors; title('x_{MMSE}: estimator');
    subplot(mplot,nplot,4); plot_spectrum_3D(y     , plotrangeS); freezeColors; title('Fourier');
    subplot(mplot,nplot,5); plot_spectrum_3D(z     , plotrangeS); freezeColors; title('Fourier');
    subplot(mplot,nplot,6); plot_spectrum_3D(x_mmse, plotrangeS); freezeColors; title('Fourier');
    pause(0.005);
    end
    
    display(['Iteration ', num2str(t), ' / ', num2str(T), ' ...']);
end
exec_time = toc;
display(['Execution time: ', num2str(exec_time), ' seconds.']);

Vout = x_mmse;

end

function U = compute_energy(ref, data, wedge)
N     = numel(ref);
refC  = real(ifftn(fftn(ref ).*wedge));
dataC = real(ifftn(fftn(data).*wedge));

U = sum(( refC(:) - dataC(:) ).^2 )/N;
end

function dataOUT = denoise(dataIN, sigma_noise)
dataOUT = bm4d(dataIN, 'Gauss', sigma_noise, 'lc', 0, 0);
end
