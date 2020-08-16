close all
clear
clc
%% Parameter

% golbal parameters 

beta = 2;                           
global dim  
global Gamma
global M
global S
global P



dim  = 4;                           % dimension in space
gamma_1=1;
gamma_2=2;
gamma=[gamma_1, gamma_2];

m_1=1;
m_2=5;
m=[m_1, m_2];

s_1=1;    %sigma
s_2=1;
s=[s_1,s_2];


p_1=1;    %P
p_2=1;
p=[p_1,p_2];





Gamma = diag(gamma);
M= diag(m);
S= diag(s);

P= diag(p);

nSample = 400;                      % number of samples           
                                    % drift coefficient                                                 
epsilon = .01;                      % regularizing coefficient                                      
h = .001;                           % time step
numSteps= 1000;                     % number of steps k, in discretization t=kh
             
%% initial condition of PDF

mean0 = rand(dim,1)';                % initial mean
sigma0 = diag(rand(dim,1)*3);     % initial variance

 
x0 = mvnrnd(mean0, sigma0,nSample);  % initial positions (generates a gaussian with mean0 and variance sigma0)

rho_0 = mvnpdf(x0, mean0,sigma0);    % Generates pdf for non-stationary case


% normalizes initial pdf in each variable


% stores all the updated locations from the governing SDE
Xupd = zeros(nSample,dim,numSteps+1);

% sets initial location
Xupd(:,:,1) = x0; 

rhoupd = zeros(nSample,numSteps+1);
rhoupd(:,1) =rho_0/sum(rho_0);

tic
for j=1:numSteps
    
     
    [driftk,GradU] = PowerDrift(Xupd(:,1:dim/2,j),Xupd(:,dim/2+1:dim,j),nSample,dim);
   
    % SDE update
  
    Xupd(:,:,j+1) = PowerEulerMaruyama(h,Xupd(:,:,j),driftk,nSample,dim); 
    
    [rhoupd(:,j+1),comptime(j),niter(j)] = FixedPointIteration(beta,epsilon,h,rhoupd(:,j),Xupd(:,:,j),Xupd(:,:,j+1),exp(-beta*PowerFraction(Xupd(:,dim/2+1:dim,j))),GradU,dim);  
    
 
end
walltime= toc;





