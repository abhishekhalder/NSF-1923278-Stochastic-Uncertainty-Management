close all; clear; clc;
%% Parameters

beta = 2; % inverse temperature

% golbal parameters 
global num_Oscillator dim Gamma M Sigma Pmech K

num_Oscillator = 2; dim  = 2*num_Oscillator; % dim = dimension of state space

gamma = rand(num_Oscillator,1); m = rand(num_Oscillator,1); sigma = rand(num_Oscillator,1); pmech = rand(num_Oscillator,1);

Gamma = diag(gamma); M = diag(m); Sigma = diag(sigma); Pmech = diag(pmech); K = rand(num_Oscillator);

% parameters for proximal recursion
nSample = 500;                      % number of samples                                                           
epsilon = .5;                     % regularizing coefficient                                      
h = 1e-3;                           % time step
numSteps= 1000;                     % number of steps k, in discretization t=kh
             
%% propagate joint PDF

% initial mean and covariance
mean0 = rand(1,dim); covariance0 = generateRandomSPD(dim);
% samples from initial joint PDF 
x0 = mvnrnd(mean0, covariance0, nSample);
% joint PDF values at the initial samples
rho_0 = mvnpdf(x0, mean0, covariance0);

% stores all the updated states from the governing SDE
Xupd = zeros(nSample,dim,numSteps+1);
% sets initial state
Xupd(:,:,1) = x0; 
% stores all the updated joint PDF values
rhoupd = zeros(nSample,numSteps+1);
% sets initial PDF values
rhoupd(:,1) = rho_0/sum(rho_0);

tic
for j=1:numSteps
    
     
    [driftk,GradU] = PowerDrift(Xupd(:,1:num_Oscillator,j),Xupd(:,num_Oscillator+1:dim,j),nSample);
   
    % SDE update for state
    Xupd(:,:,j+1) = PowerEulerMaruyama(h,Xupd(:,:,j),driftk,nSample,num_Oscillator); 
    % proximal update for joint PDF
    [rhoupd(:,j+1),comptime(j),niter(j)] = FixedPointIteration(beta,epsilon,h,rhoupd(:,j),Xupd(:,:,j),Xupd(:,:,j+1),PowerFraction(beta,Xupd(:,num_Oscillator+1:dim,j)),GradU,dim);  
    
 
end
walltime = toc

%% plots
set(0,'defaulttextinterpreter','latex')
figure(1)
semilogy(comptime, 'LineWidth', 2)
xlabel('Physical time $t=kh$','FontSize',20)
ylabel('Computational time','FontSize',20)



