function [rho_next,comptime,k] = FixedPointIteration(beta,epsilon,h,rho_prev,D1,D2,ksi,GradU,dim)

% tolerance
tol = 1e-3;   

% max number of iterations for k                                                    
maxiter = 300; 

nSample = length(D1);

% xi =  D1(:,1:dim/2); 
% xiprime = D2(:,1:dim/2);   
% eta = D1(:,dim/2+1:dim); 
% etaprime= D2(:,dim/2+1:dim);
% 
% C_1 = pdist2(etaprime,eta-(h*GradU),'squaredeuclidean');
% 
% C_2 = pdist2((xiprime-xi)/h,(eta+etaprime)/2,'squaredeuclidean');
% 
% CC = C_1+C_2;
 
transcost = @(q,p,qprime,pprime,gradu)  vecnorm( ((qprime-q)/h)-((pprime+p)/2) ).^2 +vecnorm((pprime-p +h*gradu)).^2;
for i = 1:nSample
    
    for j = 1:nSample
    
        C(i,j) = transcost(D1(i,1:dim/2),D1(i,dim/2+1:dim),D2(j,1:dim/2),D2(j,dim/2+1:dim),GradU(i,:));
   
    end
end
 
% exponential of the cost matrix
G = exp((-C)/(2*epsilon)); 
%psi = @(t) ((t.^4)/4)-((t.^2)/2) ;                      

lambda_1 = rand(nSample,1);

%lambda_1 = lambda_1/sum(lambda_1);
% initial conditions 
z0 = exp((h*lambda_1)/epsilon);    
% initial conditions for y and z
z = [z0,zeros(nSample,maxiter-1)];  


y = [rho_prev./(G*z0),zeros(nSample,maxiter-1)];

Xi = (ksi)/exp(1);


k = 1;
tic;
while k<maxiter 
 
    z(:,k+1) = (Xi./(G'*y(:,k))).^(1/(((beta*epsilon)/(h))+1));
    
    y(:,k+1) = rho_prev ./ (G*z(:,k+1));

    if (norm(z(:,k+1)-z(:,k))<tol && norm(y(:,k+1)-y(:,k))<tol)
         break;
     else
         k = k+1;
     end
end
k;
comptime = toc;

rho_next = z(:,k).*(G'*y(:,k));






    



      





    



      
