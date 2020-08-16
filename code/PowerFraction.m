function P = PowerFraction(eta)

global Gamma
global M

n = length(eta);

F = @(x) .5*x*(Gamma/M)*x';

for i = 1:n    
    
    P(i,:) = F(eta(i,:));
    
end