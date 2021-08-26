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

% parameters for the IEEE 14 bus system
Y_reduced_imag = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_Y_reduced_imag_part.csv');
Y_reduced_real = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_Y_reduced_real_part.csv');
% reduced admittance matrix
Y_reduced = complex(Y_reduced_real,Y_reduced_imag);

gen_param = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_gen_parameters.csv');
gen_bus_idx = gen_param(:,1);

num_Oscillator = numel(gen_bus_idx); dim  = 2*num_Oscillator;

% generator bus voltage
E_mag = bus_init(gen_bus_idx,2);
E_phase = bus_init(gen_bus_idx,3);
E = E_mag.*exp(1i*E_phase);

I_reduced_mag = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_I_red_magnitude.csv');
I_reduced_mag = (I_reduced_mag(2,2:end))';
I_reduced_phase = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_I_red_angle.csv');
I_reduced_phase = (I_reduced_phase(2,2:end))';

I_reduced = I_reduced_mag.*exp(1i*I_reduced_phase);

% select the time window from one day time-varying power data 
% at 1 sec resolution
initial_sec = 50000; final_sec = 51000;
% get the power trajectories within that time window
P_mech_coarse = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_P_mech_in_per_unit.csv');
P_mech_coarse = (P_mech_coarse(initial_sec:1:final_sec,2:end));
P_load_coarse = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_P_load_at_genbuses.csv');
P_load_coarse = (P_load_coarse(initial_sec:1:final_sec,2:end));

h = 1e-3; % time step
t_vec_coarse = initial_sec:1:final_sec; % time vector with 1 sec step-size
t_vec = initial_sec:h:final_sec; % time vector with h sec step-size

%initialize
P_mech = zeros(numel(t_vec),num_Oscillator);
P_load = zeros(numel(t_vec),num_Oscillator);
% interpolate the time varying powers to a finer time grid
figure
color_cell = {'r','m','b','c','g'};
for j=1:size(P_mech_coarse,2)    
    P_mech(:,j) = interp1(t_vec_coarse,P_mech_coarse(:,j),t_vec,'spline');
    P_load(:,j) = interp1(t_vec_coarse,P_load_coarse(:,j),t_vec,'spline');

    plot(t_vec,P_mech(:,j),'-','color',color_cell{j},'LineWidth',2)    
    xlabel('$t$ [s]','FontSize', 30,'Interpreter','latex')
    hold on
    plot(t_vec,P_load(:,j),'--','color',color_cell{j},'LineWidth',2)    
    xlabel('$t$ [s]','FontSize', 30,'Interpreter','latex')
    hold on
end
set(gca,'FontSize',30)
legend('$P_{1}^{\rm{mech}}$','$P_{1}^{\rm{load}}$','$P_{2}^{\rm{mech}}$','$P_{2}^{\rm{load}}$','$P_{3}^{\rm{mech}}$','$P_{3}^{\rm{load}}$','$P_{4}^{\rm{mech}}$','$P_{4}^{\rm{load}}$','$P_{5}^{\rm{mech}}$','$P_{5}^{\rm{load}}$','NumColumns',5,'FontSize', 30,'Interpreter','latex','location','best');

% effective power input
P = P_mech' - P_load' - ((E_mag.^2).*diag(Y_reduced_real)) + real(E.*(conj(I_reduced)));
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
%numSteps = 1e3;                    % number of steps k, in discretization t=kh
cc = 1e7;

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
 
invpsi = kron(eye(num_Oscillator),Sigma/M);
 
sumcov_prox = zeros(dim,dim); sumcov_mc = zeros(dim,dim);

for ii=1:num_Oscillator
    xi_0(:,ii) = wrapTo2PiMSigma(m(ii)/sigma(ii)*theta_0(:,ii),m(ii),sigma(ii)); 
end
eta_0 = (psi_lower_diag*omega_0')';
 
xi_eta_0 = [xi_0,eta_0];
 
rho_xi_eta_0 = rho_theta_omega_0/(prod(m./sigma));
 
% stores all the updated (states from the governing SDE)
xi_eta_upd = zeros(nSample,dim,length(t_vec));
 
theta_upd = zeros(nSample,num_Oscillator,length(t_vec));
omega_upd = zeros(nSample,num_Oscillator,length(t_vec));
theta_omega_upd = zeros(nSample,dim,length(t_vec));
 
% sets initial state
xi_eta_upd(:,:,1) = xi_eta_0; 
theta_upd(:,:,1) = theta_0;
omega_upd(:,:,1) = omega_0;

% stores all the updated joint PDF values
rho_xi_eta_upd = zeros(nSample,length(t_vec));
% sets initial PDF values
rho_xi_eta_upd(:,1) = rho_xi_eta_0/sum(rho_xi_eta_0);
mean_prox(1,:) = sum(xi_eta_upd(:,:,1).*rho_xi_eta_upd(:,1))';
 