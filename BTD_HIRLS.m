function [A,B,C,err,NMSE,Rest,Lest,k] = BTD_HIRLS(T,Atrue,Btrue,Ctrue,Ltrue,lambda,MaxIter,tol,R,L,A0,B0,C0)
%
% BTD_HIRLS     Implementing the BTD-HIRLS algorithm
%
% Usage:        [A,B,C,err,NMSE,Rest,Lest,k] = BTD_HIRLS(T,Atrue,Btrue,Ctrue,lambda,N,tol,R,L,A0,B0,C0)
%
% Input variables:
%               T:          Tensor
%               Atrue:      True A factor
%               Btrue:      True B factor
%               Ctrue:      True C factor (with Rtrue columns)
%               Ltrue:      True L_r's (Rtrue x 1 vector)
%               lambda:     Regularization parameter
%               MaxIter:    Maximum number of iterations
%               tol:        Tolerance
%               R,L:        Overestimated ranks
%               A0,B0,C0:   Initial factor values (optional)
% Output variables:
%               A,B,C:      Factor estimates
%               err:        Squared relative reconstruction error sequence
%               NMSE:       Sequence of NMSEs over blocks
%               Rest:       Sequence of R estimates
%               Lest:       Sequence of estimated L_r's
%               k:          Number of iterations required
%
% See A. A. Rontogiannis, E. Kofidis, and P. V. Giampouras, "Block-Term
% Tensor Decomposition: Model Selection and Computation", IEEE JSTSP, Apr. 2021
%
% Tensorlab (https://tensorlab.net/) is employed.
%
% Last update:  2 Dec. 2021
%

% Tensor dimensions
[I,J,K] = size(T);

% Useful products
JK = J*K;
KI = K*I;
LR = L*R;                               % initial value of L*R

% Modal unfoldings (transposed)
T1 = tens2mat(T,[3 2],1);
T2 = tens2mat(T,[1 3],2);
T3 = tens2mat(T,[2 1],3);
% Energy of the tensor
normT2 = frob(T,'squared');

% True value of R
Rtrue = size(Ctrue,2);

% Initialization
if (nargin < 11)                        % Random initialization if no initial values are given
    A = randn(I,LR); B = randn(J,LR); C = randn(K,R);
else
    A = A0; B = B0; C = C0;             % Could also initialize A, B only and compute C based on their values
end
% Smoothing constant
eta2 = 1e-10;
% Threshold for pruning
thr = 1e-4;
% Relative reconstruction error sequence
err = zeros(1,MaxIter);
err_prev = 1;
% Sequence of NMSEs per block
NMSE = zeros(1,MaxIter);
% R  and L_r estimates
Rest = zeros(1,MaxIter);
Lest = zeros(Rtrue,MaxIter);
% Block Khatri-Rao product of B and C
P = zeros(JK,LR);

%
% Perform iterations
%
k = 1;                                  % First iteration
rerr = 1;                               % Large relative error to start with
while rerr > tol

    % Compute D matrix
    [d1,d2] = update_Ds(A,B,C,eta2);
    D1 = diag(d1);
    D2 = diag(d2);
    D = kron(D1,eye(L))*D2;

    % Khatri-Rao product of B and C (=P)
    for r = 1:R
        P(:,(r-1)*L+1:r*L) = kron(B(:,L*(r-1)+1:L*r),C(:,r));
    end
    % Update A
    A = factor_update(P,T1',D,lambda);

    % Khatri-Rao product of C and A (=Q matrix)
    Q = zeros(KI,LR);
    for r = 1:R
        Q(:,(r-1)*L+1:r*L) = kron(C(:,r),A(:,L*(r-1)+1:L*r));
    end
    % Update B
    B = factor_update(Q,T2',D,lambda);

    % Columnwise Khatri-Rao product of A and B, times the block diagonal
    % matrix of 1_L's (=S matrix)
    onesL = kron(eye(R),ones(L,1));
    S = kr(A,B)*onesL;
    % Update C
    C = factor_update(S,T3',D1,lambda);

    % Find the columns of C with negligible energy
    %
    % First normalize C to unit Frobenius norm
    Cnormalized = C/frob(C);
    % and then threshold the energies of its columns
    temp = sum(Cnormalized.^2);
    Ind = temp<thr;             % indices of redundant columns
    % Eliminate (prune) negligible c_r's
    C(:,Ind) = [];
    % Eliminate (prune) corresponding A_r blocks
    A = reshape(A,I,L,R);   % in tensor form, with the A_r blocks at its frontal slices
    A(:,:,Ind) = [];
    % Similarly eliminate (prune) B_r blocks
    B = reshape(B,J,L,R);
    B(:,:,Ind) = [];

    % R estimate = number of non-negligible columns of C
    R = sum(Ind == 0);
    Rest(k) = R;
    LR = L*R;                   % new L*R

    % Remove permutation ambiguities and compute NMSE over blocks
    [~,~,NMSE(k),Lest(:,k)] = btderr(A,Atrue,B,Btrue,C,Ctrue,Ltrue,thr);

    % Reshape A and B back to matrices after block pruning
    A = reshape(A,I,LR);
    B = reshape(B,J,LR);

    % Compute the updated Khatri-Rao product of B and C
    P = zeros(JK,LR);
    for r = 1:R
        P(:,(r-1)*L+1:r*L) = kron(B(:,L*(r-1)+1:L*r),C(:,r));
    end
    % Estimated tensor (its mode-1 transposed unfolding)
    T1_est = P*A';
    % Normalized reconstruction error
    err(k) = frob(T1-T1_est,'squared')/normT2;
    % Relative difference of normalized reconstruction errors
    rerr = abs(err(k)-err_prev)/err(k);
    err_prev = err(k);

    % Count iterations
    k = k+1;
    % Do not iterate more than MaxIter times
    if k > MaxIter
        break;
    end

end     %while
k = k-1;
end     % end of function

%--------------------------------------------------------------------------

function M = factor_update(MM,MT,DM,lambda)
%
% FACTOR_UPDATE     Updating a BTD factor
%
% Usage:        M = factor_update(MM,MT,DM,lambda)
%
% Input variables:
%   MM:         Khatri-Rao product of the other factors
%   MT:         Corresponding tensor unfolding
%   DM:         Corresponding reweighting matrix
%   lambda:     Regularization parameter
% Output variable:
%   M:          Updated BTD factor
%
% Last update:  2 Dec. 2021
%

M = (MT*MM)/(MM'*MM+lambda*DM);    % This admits simplification; ignored here

end

%--------------------------------------------------------------------------

function [d1,d2] = update_Ds(A,B,C,eta2)
%
% UPDATE_DS     Compute re-weighting matrices
%
% Usage:        [d1,d2] = update_Ds(A,B,C,eta2)
%
% Input variables:
%   A,B,C:  Current values of BTD factors
%   eta2:   Smoothing constant
% Output variables:
%   d1,d2:  Diagonals of the D1, D2 reweighting matrices
%
% Last update:  2 Dec. 2021
%

R = size(C,2);
L = size(A,2)/R;
eta2s = repmat(eta2,1,R);
eta2sL = repmat(eta2s,1,L);
AB2 = sum([[A; B].^2; eta2sL]);
d21 = sqrt(AB2);
d2 = 1./d21;
d1 = 1./sqrt(sum([d21*kron(eye(R),ones(L,1)); sum([C.^2; eta2s])]));

end