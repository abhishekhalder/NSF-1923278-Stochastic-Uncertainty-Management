clear; clc;
define_constants;


mpc = loadcase('case14.m');
results = runopf(mpc);

z = mpc.branch(:,BR_R)+i*mpc.branch(:,BR_X);
y = 1./z;
k = abs(y);
phi = angle(y);

% line parameters
lines = [mpc.branch(:,[F_BUS,T_BUS,BR_R, BR_X]), k, phi];
writematrix(lines,'line-parameters.csv');

% bus initial states (at equilibrium pt)
buses = [mpc.bus(:,[BUS_I, VM]),deg2rad(mpc.bus(:,VA))];
writematrix(buses,'bus-init.csv');

% generator initial states (at equilibrium pt)
w0 = 2*pi*60;
H = [2.64; 2.61; 3.42; 2.45; 3.59];
D = [4; 4; 4; 4; 4];
m = 2*w0.*H;
gamma = w0.*D; 
gens = [results.gen(:,GEN_BUS),results.gen(:,PG)/mpc.baseMVA, H,D,m,gamma];
writematrix(gens,'gen_parameters.csv');

% Kron Reduction
Y = makeYbus(mpc);

nbus = size(mpc.bus,1);
inner_bus_ind = find(mpc.bus(:,BUS_TYPE)==PQ);
boundary_bus_ind = setdiff(1:nbus, inner_bus_ind);

Y_red = Y(boundary_bus_ind ,boundary_bus_ind ) ...
    - Y(boundary_bus_ind, inner_bus_ind)*inv(Y(inner_bus_ind,inner_bus_ind))*Y(inner_bus_ind,boundary_bus_ind);

Y_red_real = real(Y_red);
Y_red_imag = imag(Y_red);

writematrix(Y_red_real,'Y_reduced_real_part.csv');
writematrix(Y_red_imag,'Y_reduced_imag_part.csv');

% load data
% constant power load for those at generator buses
load_at_genbus = [results.bus(boundary_bus_ind,BUS_I), results.bus(boundary_bus_ind,PD)/mpc.baseMVA];
writematrix(load_at_genbus,'P_load_at_genbuses.csv');

% constant current load for those at generator buses
V_load = results.bus(inner_bus_ind, VM) .* exp(i*deg2rad(results.bus(inner_bus_ind, VA)) );
S_load = (results.bus(inner_bus_ind,PD)+i*results.bus(inner_bus_ind,QD)) / mpc.baseMVA;
I_load = conj( S_load ./ V_load );

I_red = Y(boundary_bus_ind, inner_bus_ind)*inv(Y(inner_bus_ind,inner_bus_ind))*I_load;

% constant current load - magnitudes
writematrix(abs(I_red),'I_red_magnitude.csv');
writematrix(angle(I_red),'I_red_angle.csv');