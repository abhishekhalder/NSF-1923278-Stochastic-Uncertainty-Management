function wsquared = Wasserstein(X,Y,rho_x,rho_y)

% x,y each must be of size nsample x dimension

m = size(X,1); n = size(Y,1);

% Equality constraints
Aeq = [kron(ones(1,n),speye(m));
      kron(speye(n),ones(1,m))];

beq = [rho_x/(sum(rho_x)); rho_y/(sum(rho_y))]; % stack marginal pmf vectors  
% Cost matrix  
%[X,Y]=meshgrid(x,y); C = (X-Y).^2; % only works for univariate PDFs
C = (pdist2(X,Y)).^2;
c = reshape(C, m*n, 1); % vectorize the cost for linprog
clear C; % free some space

% solve LP
[pmf_opt,wsquared] = linprog(c,[],[],Aeq,beq,sparse(zeros(m*n,1)),Inf(m*n,1));

%OptimalCouplingMatrix = reshape(pmf_opt,m,n);