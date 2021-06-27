function w_gam_squared = EntropyRegularizedWasserstein(X,Y,rho_x,rho_y,gam)

% x,y each must be of size nsample x dimension

%[X,Y] = meshgrid(x,y);

C = (pdist2(X,Y)).^2;

M = exp(-C/gam);

OptimalCouplingMatrix_gam = SinkhornOMT(M, rho_x, rho_y);

Elementwise_log = log(OptimalCouplingMatrix_gam);
Elementwise_log(Elementwise_log == -Inf) = 0; % force log(0) = -Inf terms to zero

w_gam_squared = trace(C'*OptimalCouplingMatrix_gam) - gam*sum(sum(OptimalCouplingMatrix_gam.*Elementwise_log));