close all; clear all; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
%% Parameters
beta = 2; % inverse temperature
% golbal parameters 
global num_Oscillator dim Gamma M Sigma K P cc sigma m phi

f0 = 60; % nominal frequency (Hz)

bus_init = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/bus-init.csv');

% prefixes
% case0_norminal_
% case1_line_13_failure_
% case2_time_series_

% parameters for the IEEE 14 bus system
Y_reduced_imag = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_Y_reduced_imag_part.csv');
Y_reduced_real = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_Y_reduced_real_part.csv');
% reduced admittance matrix
Y_reduced = complex(Y_reduced_real,Y_reduced_imag);

gen_param = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_gen_parameters.csv');
gen_bus_idx = gen_param(:,1);

num_Oscillator = numel(gen_bus_idx); dim  = 2*num_Oscillator;

% generator bus voltage
E_mag = bus_init(gen_bus_idx,2);
E_phase = bus_init(gen_bus_idx,3);
E = E_mag.*exp(1i*E_phase);
% reduced current vector
%I_reduced_polar = readmatrix('I_red.csv');
%I_reduced_mag = I_reduced_polar(:,1);
%I_reduced_phase = I_reduced_polar(:,2);
I_reduced_mag = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_I_red_magnitude.csv');
I_reduced_mag = (I_reduced_mag(2,2:end))';
I_reduced_phase = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_I_red_angle.csv');
I_reduced_phase = (I_reduced_phase(2,2:end))';

I_reduced = I_reduced_mag.*exp(1i*I_reduced_phase);

%P_mech = gen_param(:,2);
P_mech = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_P_mech_in_per_unit.csv');
P_mech = (P_mech(2:end))';
P_load = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case1_line_13_failure_P_load_at_genbuses.csv');
P_load = (P_load(2:end))';

% effective power input
P = P_mech - P_load - ((E_mag.^2).*diag(Y_reduced_real)) + real(E.*(conj(I_reduced)));
% phase shift
phi = -atan(Y_reduced_real./Y_reduced_imag);
phi = phi - diag(diag(phi)); % zero the diagonal entries
% coupling
K = (diag(E_mag))*(abs(Y_reduced))*(diag(E_mag));
K = K - diag(diag(K)); % zero the diagonal entries

% generator inertia 
m = gen_param(:,4)/(2*pi*f0);
% generator daming
gamma = gen_param(:,5)/(2*pi*f0);
% generator noise coeff
s_a = 1  ; s_b = 5;
sigma = (s_b-s_a)*rand(num_Oscillator,1) + s_a;

Gamma = diag(gamma);  M = diag(m); Sigma = diag(sigma); P = diag(P);

theta0_a = 0 ; theta0_b = 2*pi; omega0_a = -.1 ; omega0_b = .1;

% parameters for proximal recursion
nSample = 1000;                      % number of samples                                                           
epsilon = 0.5;                      % regularizing coefficient                                      
h = 1e-3;                         % time step
numSteps = 1e3;                    % number of steps k, in discretization t=kh
cc = 1e7;
t_vec = h*(1:1:numSteps);
%% popagate joint PDF
 
% samples from initial joint PDF
theta_0 = (theta0_b-theta0_a)*rand(nSample,num_Oscillator) + theta0_a;

% initial mean for the angles obtained from steady state power flow
mu_theta_0 = wrapTo2Pi(E_phase);
% initial concentration parameter for Von Mises
kappa = [5; 6; 7; 4; 5];
% create initial theta PDF as product of von Mises PDF
for j=1:num_Oscillator
    
    rho_theta_0(:,j) = exp(kappa(j)*cos(2*theta_0(:,j) - mu_theta_0(j)));

end
rho_theta_0 = prod(rho_theta_0,2);

normalization_rho_theta_0 = ((2*pi)^num_Oscillator)*prod(besseli(0,kappa));

rho_theta_0 = rho_theta_0/normalization_rho_theta_0;

omega_0 =(omega0_b-omega0_a)*rand(nSample,num_Oscillator) + omega0_a;

% joint PDF values at the initial samples
rho_omega_0 = ones(nSample,1)*(1/(omega0_b-omega0_a))^num_Oscillator;
 
rho_theta_omega_0 = rho_omega_0.*rho_theta_0;
 
psi_upper_diag = M/Sigma;
 
psi_lower_diag = psi_upper_diag;
 
psi = kron(eye(num_Oscillator),M/Sigma);
 
invpsi =kron(eye(num_Oscillator),Sigma/M);
 
sumcov_prox = zeros(dim,dim); sumcov_mc = zeros(dim,dim);

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

% stores all the updated joint PDF values
rho_xi_eta_upd = zeros(nSample,numSteps+1);
% sets initial PDF values
rho_xi_eta_upd(:,1) = rho_xi_eta_0/sum(rho_xi_eta_0);
mean_prox(1,:) = sum(xi_eta_upd(:,:,1).*rho_xi_eta_upd(:,1))';
 
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
rho_theta_omega_upd(:,1) = rho_theta_omega_0;
%% MC and proximal mean vectors
mean_mc_omega  = mean(squeeze(omega_upd));

mean_mc_theta = weighted_angle_mean(theta_upd,ones(nSample,1)/nSample);
mean_mc_omega = squeeze(mean_mc_omega);
mean_mc_theta = squeeze(mean_mc_theta);

mean_mc = [mean_mc_theta;mean_mc_omega];
mean_prox = [mean_prox_theta';mean_prox_omega'];

norm_diff_mean_mc_vs_prox = sqrt(sum((mean_mc - mean_prox).^2,1))./sqrt(sum(mean_mc.^2,1));

%% plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% semilogy(t_vec, comptime, 'LineWidth', 2)
% set(gca,'FontSize',30)
% xlabel('Physical time $t=kh$ [s]','FontSize',30)
% ylabel('Computational time [s]','FontSize',30)
% % ylim([1e-3 1.2e-2])
% % YTick = [2e-3 8e-3 2e-2];
% % YTickLabels = cellstr(num2str(round(log10(YTick(:))), '10^%d'));
% grid on
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(2)
% semilogy(t_vec(4:end), norm_diff_mean_mc_vs_prox(5:end),'-k','Linewidth',2)
% set(gca,'FontSize',30)
% xlabel('Physical time $t=kh$ [s]','FontSize',30)
% ylabel('Realtive error $\frac{\|\mu_{\rm{MC}}-\mu_{\rm{Prox}}\|_{2}}{\|\mu_{\rm{MC}}\|_{2}}$','FontSize',30,'interpreter','latex')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fs  = 15;
% for i=1:dim
%     if i<=num_Oscillator
%       figure(2);
%       subplot(1,num_Oscillator,i)
%     plot(mean_mc(i,2:end),'Linewidth',1)
%     
%     hold on
%     plot(mean_prox(i,2:end),'--k','Linewidth',.5)
%    
%     xlabel('$t$','fontsize',fs,'interpreter','latex')
%     ylabel(sprintf('$\\theta_{%d}$', i),'fontsize',fs, 'Interpreter','latex','rotation',0);
%     axis tight
%     
%     else 
%      figure(3);
%      subplot(1,num_Oscillator,i-num_Oscillator)   
%     plot(mean_mc(i,2:end),'Linewidth',1)
%     
%     hold on
%     
%     plot(mean_prox(i,2:end),'--k','Linewidth',.5)
%     xlabel('$t$','fontsize',fs,'interpreter','latex')
%     ylabel(sprintf('$\\omega_{%d}$', i-num_Oscillator),'fontsize',fs, 'Interpreter','latex','rotation',0);
%     axis tight
%     end  
% end
% legend('Mean MC','Mean Proximal')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute and save marginal PDFs
nBins = 15;

t_now_vec = 0.2:0.2:1; % snapshots of interest for marginals
t_now_idx_vec = floor(t_now_vec/h);
t_now_idx_vec(1)=1;

thetalabels = {'$\theta_1$','$\theta_2$','$\theta_3$','$\theta_4$','$\theta_5$'};
omegalabels = {'$\omega_1$','$\omega_2$','$\omega_3$','$\omega_4$','$\omega_5$'};

polar_theta = 0:0.05:2*pi;
xx = cos(polar_theta); yy = sin(polar_theta);
xt = [1 0 -1 0 -0.5]; yt = [0 1 -0.64 -1 -1.25];
%figure(3)
figure('color','w')
for j=1:num_Oscillator
    for tt=1:length(t_now_idx_vec)
        % univariate omega marginals
        [omega_save1D{tt,j},marg1D_omega_save{tt,j}] = getMarginal1D(omega_upd(:,j,tt),rho_theta_omega_upd(:,tt),nBins);

        % bivariate marginals
        [theta_temp{tt,j},omega_save{tt,j},marg2D_theta_omega_temp{tt,j}] = getMarginal2D(theta_upd(:,j,tt),omega_upd(:,j,tt),rho_theta_omega_upd(:,tt),nBins);
        % make new bivariate grid     
        [theta_save{tt,j},omega_save{tt,j}] = meshgrid(linspace(0,2*pi,nBins),linspace(min(omega_save{tt,j}(:)),max(omega_save{tt,j}(:)),nBins));
        % interpolate bivariate marginal on new bivariarte grid
        marg2D_theta_omega_save{tt,j} = interp2(theta_temp{tt,j},omega_save{tt,j},marg2D_theta_omega_temp{tt,j}.PF,theta_save{tt,j},omega_save{tt,j},'spline');
        % zero the spurious negative values of the bivariate marginal resulting from interpolation
        marg2D_theta_omega_save{tt,j}(marg2D_theta_omega_save{tt,j}<0) = 0;
        
        maxMargValMatrix(tt,j) = max(max(marg2D_theta_omega_save{tt,j}));

        subplot(num_Oscillator,length(t_now_idx_vec),tt+(j-1)*length(t_now_idx_vec))
        
        siz=size(theta_save{tt,j});
        V=[cos(theta_save{tt,j}(:)) sin(theta_save{tt,j}(:)) omega_save{tt,j}(:)];
        % Face-vertex connectivity list
        numV=size(V,1);                        % total number of vertices 
        id=reshape(1:numV,siz);                % vertex indices
        F1=id(1:(siz(1)-1),1:(siz(2)-1));
        F2=id(2:(siz(1)-0),1:(siz(2)-1));
        F3=id(2:(siz(1)-0),2:(siz(2)-0));
        F4=id(1:(siz(1)-1),2:(siz(2)-0));
        F=[F1(:) F2(:) F3(:) F4(:)];
        
        G=marg2D_theta_omega_save{tt,j}(:); 
        % contourf(theta_save{tt,j},omega_save{tt,j},marg2D_theta_omega_save{tt,j});
        
        hh=patch('Faces',F,'Vertices',V,'FaceVertexCData',G,'FaceColor','interp');
        set(hh,'FaceAlpha',0.65,'EdgeColor','none')
        view([20 20])
                
%         xlabel(thetalabels{j},'FontSize',30)
%         ylabel(omegalabels{j},'FontSize',30,'Rotation',0)       
        zlabel(omegalabels{j},'FontSize',30,'Rotation',0)
        camlight('headlight'), lighting phong

        if j==1
            title(['$t=$' num2str(t_now_vec(tt))],'interpreter','latex')
        end
%         set(gca,'XTick',0:pi/2:2*pi, 'XTickLabel',{'$0$','$\frac{\pi}{2}$','$\pi$','$\frac{3\pi}{2}$','$2\pi$'})
%         set(get(gca,'XAxis'),'TickDir','out')
        
%         Tri=[F(:,[1 2 3]);F(:,[3 4 1])];
%         [~]=IsoContour({Tri V},G,10,gca);        
        hold on
        
        plot3(xx,yy,(min(omega_save{tt,j}(:)))*ones(1,numel(xx)),'-k','LineWidth',2)
        axis tight
        set(gca,'FontSize',30)
        set(gca,'XTick',[], 'YTick', [])
        text(xt,yt,(min(omega_save{tt,j}(:)))*[1.2, 0.76, 1, 1.34, 1.15],{'$0$','$\pi/2$','$\pi$','$3\pi/2$',thetalabels{j}},'FontSize',30)
        hold on
    end    
end
% plot common colorbar
left1 = 0.13; cb_bottom = 0.05; cb_width = 0.80; cb_height = 0.02;
  
cbax = axes('visible', 'off');
caxis(cbax, [0, max(max(maxMargValMatrix))]);
h = colorbar('peer', cbax, 'southoutside', ...
  'position', [left1 cb_bottom cb_width cb_height],...
  'FontSize',30,'TickLabelInterpreter','latex');

% plot 1D omega marginals
figure;
for j=1:num_Oscillator
    for tt=1:length(t_now_idx_vec)
        subplot(num_Oscillator,length(t_now_idx_vec),tt+(j-1)*length(t_now_idx_vec))
        plot(omega_save1D{tt,j},marg1D_omega_save{tt,j}.MC,'--ro','LineWidth',2)
        hold on
        plot(omega_save1D{tt,j},marg1D_omega_save{tt,j}.PF1,'-bs','LineWidth',2)
        hold on
        plot(omega_save1D{tt,j},marg1D_omega_save{tt,j}.PF2,'-kd','LineWidth',2)
        if j==1
            title(['$t=$' num2str(t_now_vec(tt))],'interpreter','latex')
        end
    end
end

% save marginal data as .txt file
for j=1:num_Oscillator
    for tt=1:length(t_now_idx_vec)
        % save univariate omega marginal data
        textfilename_omega1D = ['case1_line_13_failure_IEEE14BusGenIdx' num2str(j) 'omega1Dt' num2str(t_now_vec(tt)) '.txt'];
        dlmwrite(textfilename_omega1D, omega_save1D{tt,j},'delimiter','\t','precision','%f');
        
        textfilename_marg1D = ['case1_line_13_failure_IEEE14BusGenIdx' num2str(j) 'marg1Dt' num2str(t_now_vec(tt)) '.txt'];
        dlmwrite(textfilename_marg1D, marg1D_omega_save{tt,j}.MC,'delimiter','\t','precision','%f');
        
        % save bivariate marginal data
        textfilename_theta2D = ['case1_line_13_failure_IEEE14BusGenIdx' num2str(j) 'theta2Dt' num2str(t_now_vec(tt)) '.txt'];
        dlmwrite(textfilename_theta2D, theta_save{tt,j},'delimiter','\t','precision','%f');
    
        textfilename_omega2D = ['case1_line_13_failure_IEEE14BusGenIdx' num2str(j) 'omega2Dt' num2str(t_now_vec(tt)) '.txt'];
        dlmwrite(textfilename_omega2D, omega_save{tt,j},'delimiter','\t','precision','%f');
    
        textfilename_marg2D = ['case1_line_13_failure_IEEE14BusGenIdx' num2str(j) 'marg2Dt' num2str(t_now_vec(tt)) '.txt'];
        dlmwrite(textfilename_marg2D, marg2D_theta_omega_save{tt,j},'delimiter','\t','precision','%f');
    end    
end

%% save simulation time data
textfilename = 'case1_line_13_failure_TimeSyntheticIEEE14bus.txt';
dlmwrite(textfilename, t_vec,'delimiter','\t','precision','%.64f');

textfilename = 'case1_line_13_failure_ComptimeSytheticIEEE14bus.txt';
dlmwrite(textfilename, comptime,'delimiter','\t','precision','%.64f');