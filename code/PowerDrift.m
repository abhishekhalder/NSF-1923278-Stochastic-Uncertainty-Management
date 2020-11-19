function [driftk,GradU] = PowerDrift(xi,eta,nSample)

global num_Oscillator phi Gamma M Sigma Pmech K sigma  m
 
GradF_eta = ((Gamma/M)*eta')';

GradU_2 = -repmat((diag(Pmech/Sigma))',nSample,1);
% 
% for l=1:nSample
% 
%     GradU_1(l,:)=K*(Sigma\sin(diag(diag(Sigma/M)*xi(l,:)))-(diag(Sigma/M)*xi(l,:)*ones(num_Oscillator,1)));
%   
% end

for l = 1:nSample
    
for i = 1:num_Oscillator
    
    sum = 0;
    for j =1:num_Oscillator 
       
        sum = sum +( K(i,j)/sigma(i) )*sin((sigma(i)/m(i))*xi(l,i)-(sigma(j)/m(j))*xi(l,j)+phi(i,j));
    end
    
    GradU_1(l,i) = sum;
end

end

GradU_xi = GradU_1 + GradU_2;

driftk =[eta, -GradF_eta-GradU_xi];

GradU = GradU_xi;