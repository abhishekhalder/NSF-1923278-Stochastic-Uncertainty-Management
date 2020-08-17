function [driftk,GradU] = PowerDrift(xi,eta,nSample)

global num_Oscillator dim Gamma M Sigma Pmech K
 
GradF_eta = ((Gamma/M)*eta')';

GradU_2 = repmat((diag(Pmech/Sigma))',nSample,1);


for l=1:nSample

    summ = 0;
    
    for k = 1:num_Oscillator
        
        for j = 1:num_Oscillator
         
            summ = summ + K(k,j)/Sigma(k,k)*sin(Sigma(k,k)/M(k,k)*xi(l,k)-Sigma(j,j)/M(j,j)*xi(l,j));
  
        end
       GradU_1(l,k) = summ;
    end
    
end

    GradU_xi = GradU_1 + GradU_2;

    driftk =[eta, -GradF_eta-GradU_xi];

    GradU = GradU_xi;
   
end



