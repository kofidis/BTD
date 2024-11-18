function [foundR,foundL_r,NMSE,Lest] = btderr(A,Atrue,B,Btrue,C,Ctrue,Ltrue,thr)
%
% BTDERR    Calculate errors in BTD model selection and computation
%
% Usage: [foundR,foundL_r,NMSE,Lest] = btderr(A,Atrue,B,Btrue,C,Ctrue,Ltrue,thr)
%
% Input variables:
%           A:          Estimated A factor (shaped as a 3-way tensor: I x L x Rest)
%           Atrue:      True A factor (cell array)
%           B:          Estimated B factor (shaped as a 3-way tensor: J x L x Rest)
%           Btrue:      True B factor (cell array)
%           C:          Estimated C factor (K x Rest)
%           Ctrue:      True C factor (K x Rtrue)
%           Ltrue:      Rtrue x 1 vector containing the true L_r's
%           thr:        Threshold (prune columns with smaller energy)
% Output variables:
%           foundR:     1 if R was found, 0 otherwise
%           foundL_r:   1's / 0's for correctly / incorrectly estimated
%                       L_r's. All zeros if R was not correctly estimated
%           NMSE:       NMSE over blocks
%           Lest:       Estimated L_r's
%
% Tensorlab (https://tensorlab.net/) is employed.
%
% Last update:  26 Feb. 2022
%

% True R = number of columns in Ctrue
Rtrue = size(Ctrue,2);
% Deduce I,J,K from the sizes of A, B, C
I = size(Atrue{1},1);
J = size(Btrue{1},1);
K = size(Ctrue,1);
IJK = I*J*K;
% Estimated R = number of columns of C
Rest = size(C,2);
% Will contain the estimated L_r's
Lest = zeros(Rtrue,1);
% Will contain the A_r and B_r blocks
Aest = cell(Rest,1);
Best = cell(Rest,1);

% Check if R was correctly found
foundR = Rest == Rtrue;

% Will contain the vectorized block terms in their columns
block_true = zeros(IJK,Rtrue);
block_est = zeros(IJK,Rest);
% True block terms
for i = 1:Rtrue
    temp = outprod(Atrue{i}*Btrue{i}.',Ctrue(:,i));
    block_true(:,i) = temp(:);
end
% Estimated block terms
for i = 1:Rest
    Aest{i} = A(:,:,i);
    Best{i} = B(:,:,i);
    temp = outprod(Aest{i}*Best{i}.',C(:,i));
    block_est(:,i) = temp(:);
end
% Compute the Frobenius distances between estimated and true block terms
dist = zeros(Rest,Rtrue);
for i = 1:Rest
    for j = 1:Rtrue
        dist(i,j) = frob(block_est(:,i)-block_true(:,j));
    end
end
% and solve the corresponding linear assignment problem
match = matchpairs(dist,10000);
% Calculate NMSE
R = min(Rest,Rtrue);
if R < Rtrue    % too few terms
    NMSE = 1;
else            % enough terms
    NSE = 0;
    for i = 1:size(match,1)
        A_r = A(:,:,match(i,1));
        % normalize to unit Frobenius norm
        A_r_n = A_r/frob(A_r);
        B_r = B(:,:,match(i,1));
        B_r_n = B_r/frob(B_r);
        Anorm2s = sum(A_r_n.^2);
        Bnorm2s = sum(B_r_n.^2);
        % L_r estimate (computed from A_r and/or B_r)
        Lest(match(i,2)) = sum(Anorm2s>=thr | Bnorm2s>=thr);
        % Sum of relative errors per block (see the definition of the NMSE
        % in the paper)
        NSE = NSE+frob(outprod(A_r*B_r.',C(:,match(i,1)))-outprod(Atrue{match(i,2)}*Btrue{match(i,2)}.',Ctrue(:,match(i,2))),'squared') ...
            /frob(outprod(Atrue{match(i,2)}*Btrue{match(i,2)}.',Ctrue(:,match(i,2))),'squared');
    end
    NMSE = NSE/size(match,1);
end
% If R was found, check the accuracy of the L_r estimates
if foundR
    foundL_r = Lest == Ltrue;
else % Consider the L_r estimates only if R was correctly recovered
    foundL_r = zeros(Rtrue,1);
end

end % function