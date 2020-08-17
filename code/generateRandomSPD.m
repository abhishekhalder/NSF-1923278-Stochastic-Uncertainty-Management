function A = generateRandomSPD(dim)
% Generate a dense n x n symmetric, positive definite matrix

A = rand(dim);

A = 0.5*(A+A'); % symmetrize

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding dim*I
A = A + dim*eye(dim);