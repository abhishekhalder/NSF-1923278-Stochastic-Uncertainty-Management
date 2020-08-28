function exp_neg_beta_F_eta = PowerFraction(beta,eta)

global Gamma M

F_eta = (0.5*dot(eta',(Gamma/M)*eta'))';

exp_neg_beta_F_eta = exp(-beta*F_eta);