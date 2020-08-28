function [rho_next,comptime,k] = FixedPointIteration(beta,epsilon,h,rho_prev,OldState,NewState,ksi,GradU,dim)

% tolerance
tol = 1e-3;   

% max number of iterations for k                                                    
maxiter = 300; 

nSample = length(OldState);

xi =  OldState(:,1:dim/2); 
xiprime = NewState(:,1:dim/2);   
eta = OldState(:,dim/2+1:dim); 
etaprime = NewState(:,dim/2+1:dim);

C_1 = pdist2(eta-(h*GradU),etaprime,'squaredeuclidean');

C_2 = 12*pdist2( -eta/2 - xi/h,-(-etaprime/2 + xiprime/h ),'squaredeuclidean');

C = C_1+C_2;
%  
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