close all; clear; clc;
%% Parameters
beta = 2; % inverse temperature
% golbal parameters 
global num_Oscillator dim Gamma M Sigma K Pmech cc sigma m phi

num_Oscillator = 20; dim  = 2*num_Oscillator; % dim = dimension of state space

f0 = 50; % Hz
g_a = 20 ; g_b = 30; m_a = 2; m_b = 12 ; s_a = 1  ; s_b = 5;
p_a = 0; p_b = 10 ;  k_a =.7 ; k_b = 1.2;

tp_a = 0; 
tp_b = 0.25;

gamma = ((g_a-g_b)*rand(num_Oscillator,1) + g_a)/(2*pi*f0);
m = ((m_b-m_a)*rand(num_Oscillator,1) + m_a)/(2*pi*f0);
sigma = (s_b-s_a)*rand(num_Oscillator,1) + s_a;

phi = atan((tp_b-tp_a)*rand(num_Oscillator) + tp_a);
phi = phi - diag(diag(phi)); % zero the diagonal entries

pmech = (p_b-p_a)*rand(num_Oscillator,1) + p_b;

Gamma = diag(gamma);  M = diag(m); Sigma = diag(sigma); Pmech = diag(pmech);

K = (k_b-k_a)*rand(num_Oscillator)+ k_a;
K = K - diag(diag(K)); % zero the diagonal entries

theta0_a = 0 ; theta0_b = 2*pi; omega0_a = -.1 ; omega0_b = .1;

% parameters for proximal recursion
nSample = 1000;                      % number of samples                                                           
epsilon = 0.5;                      % regularizing coefficient                                      
h = 1e-3;                         % time step
numSteps= 1000;                    % number of steps k, in discretization t=kh
cc = 1e7;
t_vec = h*(1:1:numSteps);
%% propagate joint PDF
% samples from initial joint PDF 
theta_0 = (theta0_b-theta0_a)*rand(nSample,num_Oscillator) + theta0_a;

rho_theta_0 = ones(nSample,1)*(1/(theta0_b-theta0_a))^num_Oscillator;

omega_0 =(omega0_b-omega0_a)*rand(nSample,num_Oscillator) + omega0_a;

% joint PDF values at the initial samples
rho_omega_0 = ones(nSample,1)*(1/(omega0_b-omega0_a))^num_Oscillator;

rho_theta_omega_0 = rho_omega_0.*rho_theta_0;

psi_upper_diag = M/Sigma;

psi_lower_diag = psi_upper_diag;

psi = kron(eye(num_Oscillator),M/Sigma);

invpsi =kron(eye(num_Oscillator),Sigma/M);

for ii=1:num_Oscillator
    xi_0(:,ii) = wrapTo2PiMSigma(m(ii)/sigma(ii)*theta_0(:,ii),m(ii),sigma(ii));
end

eta_0 = (psi_lower_diag*omega_0')';

xi_eta_0 = [xi_0,eta_0];

rho_xi_eta_0 = rho_theta_omega_0/(prod(m./sigma));

% stores all the updated (states from the governing SDE
xi_eta_upd = zeros(nSample,dim,numSteps+1);

theta_upd = zeros(nSample,num_Oscillator,numSteps+1);

omega_upd = zeros(nSample,num_Oscillator,numSteps+1);

theta_omega_upd = zeros(nSample,dim,numSteps+1);

% sets initial state
xi_eta_upd(:,:,1) = xi_eta_0; 
theta_upd(:,:,1) = theta_0;
omega_upd(:,:,1) = omega_0;
theta_omega_upd(:,:,1) = [theta_0,omega_0];

% stores all the updated joint PDF values
rho_xi_eta_upd = zeros(nSample,numSteps+1);
rho_theta_omega_upd = zeros(nSample,numSteps+1);
% sets initial PDF values
rho_xi_eta_upd(:,1) = rho_xi_eta_0/sum(rho_xi_eta_0);

rho_theta_omega_upd(:,1) = rho_xi_eta_upd(:,1)*(prod(m./sigma));

mean_prox_omega(1,:) = sum(omega_upd(:,:,1).*rho_theta_omega_upd(:,1))/sum(rho_theta_omega_upd(:,1));  
    
mean_prox_theta(1,:) = weighted_angle_mean(theta_upd(:,:,1),rho_theta_omega_upd(:,1));

tic;
for j=1:numSteps 
    
   [drift_j,GradU] = PowerDrift(xi_eta_upd(:,1:num_Oscillator,j),xi_eta_upd(:,num_Oscillator+1:dim,j),nSample);
    
    % SDE update for state
    xi_eta_upd(:,:,j+1) = PowerEulerMaruyama(h,xi_eta_upd(:,:,j),drift_j,nSample,num_Oscillator);
    
    theta_upd(:,:,j+1) = wrapTo2Pi((psi_upper_diag\xi_eta_upd(:,1:num_Oscillator,j+1)')');
    
    omega_upd(:,:,j+1) = (psi_lower_diag\xi_eta_upd(:,num_Oscillator+1:dim,j+1)')';
        
  %proximal update for joint PDF
   [rho_xi_eta_upd(:,j+1),comptime(j),niter(j)] = FixedPointIteration(beta,epsilon,h,rho_xi_eta_upd(:,j),xi_eta_upd(:,:,j),xi_eta_upd(:,:,j+1),PowerFraction(beta,xi_eta_upd(:,num_Oscillator+1:dim,j)),GradU,dim);  
    
   rho_theta_omega_upd(:,j+1) = rho_xi_eta_upd(:,j+1)*(prod(m./sigma));

   mean_prox_omega(j+1,:) = sum(omega_upd(:,:,j+1).*rho_theta_omega_upd(:,j+1))/sum(rho_theta_omega_upd(:,j+1));  
    
   mean_prox_theta(j+1,:) = weighted_angle_mean(theta_upd(:,:,j+1),rho_theta_omega_upd(:,j+1));
end

toc

mean_mc_omega  = mean(squeeze(omega_upd));

mean_mc_theta = weighted_angle_mean(theta_upd,ones(nSample,1)/nSample);
mean_mc_omega = squeeze(mean_mc_omega);
mean_mc_theta = squeeze(mean_mc_theta);

mean_mc = [mean_mc_theta;mean_mc_omega];
mean_prox = [mean_prox_theta';mean_prox_omega'];

norm_diff_mean_mc_vs_prox = sqrt(sum((mean_mc - mean_prox).^2,1))./sqrt(sum(mean_mc.^2,1));

%% plots
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure(1)
semilogy(t_vec, comptime, 'LineWidth', 2)
set(gca,'FontSize',30)
xlabel('Physical time $t=kh$ [s]','FontSize',30)
ylabel('Computational time [s]','FontSize',30)
ylim([1e-3 1.2e-2])
YTick = [2e-3 8e-3 2e-2];
YTickLabels = cellstr(num2str(round(log10(YTick(:))), '10^%d'));
grid on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs  = 15;
for i=1:dim
    if i<=num_Oscillator
      figure(2);
      subplot(1,num_Oscillator,i)
      plot(mean_mc(i,1:end),'Linewidth',2)
    
      hold on
      plot(mean_prox(i,1:end),'--k','Linewidth',2)

      xlabel('$t$','fontsize',fs,'interpreter','latex')
      ylabel(sprintf('$\\theta_{%d}$', i),'fontsize',fs, 'Interpreter','latex','rotation',0);
      axis tight
    
    else 
     figure(3);
     subplot(1,num_Oscillator,i-num_Oscillator)   
     plot(mean_mc(i,1:end),'Linewidth',2)
    
     hold on
    
     plot(mean_prox(i,1:end),'--k','Linewidth',2)
     xlabel('$t$','fontsize',fs,'interpreter','latex')
     ylabel(sprintf('$\\omega_{%d}$', i-num_Oscillator),'fontsize',fs, 'Interpreter','latex','rotation',0);
     axis tight
    end 
end
legend('Mean MC','Mean Proximal')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4)
semilogy(t_vec(4:end), norm_diff_mean_mc_vs_prox(5:end),'-k','Linewidth',2)
set(gca,'FontSize',30)
xlabel('Physical time $t=kh$ [s]','FontSize',30)
ylabel('Realtive error $\frac{\|\mu_{\rm{MC}}-\mu_{\rm{Prox}}\|_{2}}{\|\mu_{\rm{MC}}\|_{2}}$','FontSize',30,'interpreter','latex')


%% save simulation data
textfilename = 'TimeSynthetic.txt';
dlmwrite(textfilename, t_vec,'delimiter','\t','precision','%.64f');

textfilename = 'ComptimeSythetic.txt';
dlmwrite(textfilename, comptime,'delimiter','\t','precision','%.64f');

textfilename = 'RelErrMeanVectorMCvsProxSythetic.txt';
dlmwrite(textfilename, norm_diff_mean_mc_vs_prox,'delimiter','\t','precision','%.64f');