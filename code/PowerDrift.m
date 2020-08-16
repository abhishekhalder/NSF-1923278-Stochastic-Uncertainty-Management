function [driftk,GradU] = PowerDrift(zeta,eta,nSample,dim)

global Gamma
global M
global P
global S



%dzeta = @(x)[x(1),x(2)];
 
GradF_eta = ((Gamma/M)*eta')';

GradU_2=repmat([P(1,1)/S(1,1),P(2,2)/S(2,2)],nSample,1);

K=rand(2,2);
F=zeros(nSample,dim/2,1);

for l=1:nSample

    summ=0;
    
    for k=1:dim/2
        
        for j=1:dim/2
         
            summ=summ+K(k,j)/S(k,k)*sin(S(k,k)/M(k,k)*zeta(l,k)-S(j,j)/M(j,j)*zeta(l,j));
  
        end
       GradU_1(l,k)=summ;
    end
    
end

GradU_zeta= GradU_1+GradU_2;

driftk =[eta, -GradF_eta-GradU_zeta];

GradU = GradU_zeta;
   
end



