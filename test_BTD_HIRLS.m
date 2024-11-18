% Simulation of BTD-HIRLS (synthetic data)
%
% See:  A. A. Rontogiannis, E. Kofidis, and P. V. Giampouras, IEEE J.
% Special Topics in Signal Process., Apr. 2021.
%
% Last update:  18 Nov. 2024
%

clear variables
close all
addpath('tensorlab_2016-03-28/')

rng('default');

%% Set parameter values

% Maximum number of iterations
MaxIter = 200;
% Stop when the relative difference of squared reconstruction errors becomes less
% than that
toler = 1e-6;
% SNR value (in dB)
SNR = 15;
% Number of (random) initializations
Ni = 5;
% Number of realizations
Nr = 20;

% True values of the ranks
Rt = 3;
Lt = [8 6 4]';

% Overestimates
R = 10;
L = 10;
LR = L*R;

% Tensor dimensions
I = 18;
J = 18;
K = 10;

% Initialize sets of output sequences
ERRS = [];      % squared relative reconstruction errors
NMSES = [];     % NMSEs over blocks
RESTS = [];     % R estimates
LESTS = [];     % L estimates
ITERS = [];     % numbers of iterations to convergence

% Initialize rank estimation success rates
foundR = 0;             % # realizations where R was correctly estimated
succL = zeros(Rt,1);    % L_r's estimation success rate

% To keep initializations
A0s = zeros(I,LR,Ni);
B0s = zeros(J,LR,Ni);
C0s = zeros(K,R,Ni);
NMSEs = zeros(Ni,1);

%% Run over realizations
for n = 1:Nr

    n

    % Built tensor
    for r = 1:Rt    % i.i.d. Gaussian
        At{r} = randn(I,Lt(r));
        Bt{r} = randn(J,Lt(r));
    end
    Ct = randn(K,Rt);
    Tt = zeros(I,J,K);
    for r = 1:Rt
        Tt = Tt+outprod(At{r}*Bt{r}',Ct(:,r));
    end
    N = randn(I,J,K);                   % noise
    sigma = 10^(-SNR/20)*frob(Tt)/frob(N);
    T = Tt+sigma*N;

    % Regularization parameter (set here, as it depends on the noise
    % power)
    lambda = 0.1*R*((I+J)*L+K)*sigma;    % "Play" with this rule of thumb

    % Random initialization(s)
    for i = 1:Ni
        A0 = randn(I,LR);
        B0 = randn(J,LR);
        C0 = rand(K,R);
        % Store these initial values
        A0s(:,:,i) = A0;
        B0s(:,:,i) = B0;
        C0s(:,:,i) = C0;
        % Run the BTD-HIRLS algorithm (till convergence; could be stopped
        % earlier)
        [~,~,~,~,NMSE,~,~,k] = BTD_HIRLS(T,At,Bt,Ct,Lt,lambda,MaxIter,toler,R,L,A0,B0,C0);
        NMSEs(i) = NMSE(k);
    end
    % Choose the best initialization
    [~,min_i] = min(NMSEs);
    A0 = A0s(:,:,min_i);
    B0 = B0s(:,:,min_i);
    C0 = C0s(:,:,min_i);
    % and use it (once more, for simplicity here) to get results
    [A,B,C,err,NMSE,Rest,Lest,k] = BTD_HIRLS(T,At,Bt,Ct,Lt,lambda,MaxIter,toler,R,L,A0,B0,C0);

    % Collect numbers of iterations to convergence
    ITERS = [ITERS k];
    % Record success in revealing ranks
    if Rest(k) == Rt
        foundR = foundR+1;
        succL = succL+(Lest == Lt);
    end
    % Collect relative reconstruction error sequences
    ERRS = [ERRS; err];
    % Collect NMSE sequences
    NMSES = [NMSES; NMSE];
    % Collect R and L_r estimates
    RESTS = [RESTS; Rest(k)];
    LESTS = [LESTS; Lest];

end     % realizations

% Rank revealing success rates
succR = foundR/Nr;
succL = succL/foundR;

%% Plot results
% Averaging over realizations
errs = mean(ERRS);
nmses = mean(NMSES);
figure(1)
subplot(321)
semilogy(errs)
xlabel('Iterations')
ylabel('RSE')
grid
subplot(322)
semilogy(nmses)
xlabel('Iterations')
ylabel('NMSE')
grid
subplot(323)
histogram(RESTS,R,'Normalization','probability')
xlabel('R values')
ylabel('Success rate')
grid
subplot(324)
% Should modify this to work for any Rt
histogram(LESTS(1,:),L,'Normalization','probability')
xlabel('L_1 values')
ylabel('Success rate')
grid
subplot(325)
histogram(LESTS(2,:),L,'Normalization','probability')
xlabel('L_2 values')
ylabel('Success rate')
grid
subplot(326)
histogram(LESTS(3,:),L,'Normalization','probability')
xlabel('L_3 values')
ylabel('Success rate')
grid