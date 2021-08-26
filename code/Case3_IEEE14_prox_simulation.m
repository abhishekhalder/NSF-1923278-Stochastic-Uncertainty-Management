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

%P_mech = gen_param(:,2);
P_mech_coarse = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_P_mech_in_per_unit.csv');
%P_mech = (P_mech(2:end))';
P_load_coarse = readmatrix('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/Data/case3_realworld_time_series_P_load_at_genbuses.csv');
%P_load = (P_load(2:end))';


