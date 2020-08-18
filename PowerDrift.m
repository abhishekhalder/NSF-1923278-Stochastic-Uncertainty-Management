function [driftk,GradU] = PowerDrift(xi,eta,nSample)

global num_Oscillator dim Gamma M Sigma Pmech K
 
GradF_eta = ((Gamma/M)*eta')';

GradU_2 = repmat((diag(Pmech/Sigma))',nSample,1);

 


for l=1:nSample


    GradU_1(l,:)=K*inv(Sigma)*sin(diag(diag(Sigma/M)*xi(l,:))-(diag(Sigma/M)*xi(l,:)*ones(num_Oscillator,1)));
  
    
    
end

    GradU_xi = GradU_1 + GradU_2;

    driftk =[eta, -GradF_eta-GradU_xi];

    GradU = GradU_xi;
   
end



