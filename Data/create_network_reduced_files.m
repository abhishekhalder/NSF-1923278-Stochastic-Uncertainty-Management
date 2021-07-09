function create_network_reduced_files(mpc,  second_series, gen_series, PD_series, QD_series, VM_series, VA_series, note)

define_constants;
nbus = size(mpc.bus,1);


nt = size(second_series,1);
ngen = size(gen_series,1);

assert(nt==size(gen_series,2) ); assert(ngen==size(gen_series,1));
assert(nt==size(PD_series,2) ); assert(nbus==size(PD_series,1) );
assert(nt==size(QD_series,2) ); assert(nbus==size(QD_series,1) );
assert(nt==size(VM_series,2) ); assert(nbus==size(VM_series,1) );
assert(nt==size(VA_series,2) ); assert(nbus==size(VA_series,1) );

z = mpc.branch(:,BR_R)+i*mpc.branch(:,BR_X);
y = 1./z;
k = abs(y);
phi = angle(y);

% line parameters
lines = [mpc.branch(:,[F_BUS,T_BUS,BR_R, BR_X]), k, phi];
line_header = {'from-bus','no-bus','resistance-R','reactance-X','admittance-y-magnitude','admittance-y-angle'};
write_csv_with_header([note,'_line_parameters.csv'],line_header,lines);

% bus initial states (at equilibrium pt)
buses = [mpc.bus(:,[BUS_I, VM]),deg2rad(mpc.bus(:,VA))];
writematrix(buses,'bus-init.csv');

% generator initial states (at equilibrium pt)
w0 = 2*pi*60;
H = [2.64; 2.61; 3.42; 2.45; 3.59];
D = [4; 4; 4; 4; 4];
m = 2*w0.*H;
gamma = w0.*D; 

gens = [mpc.gen(:,GEN_BUS), H,D,m,gamma];
gen_header = {'BusNum(in 14 bus)','H','D','m','gamma'};
write_csv_with_header([note,'_gen_parameters.csv'],gen_header,gens);

% Kron Reduction
Y = makeYbus(mpc);


inner_bus_ind = find(mpc.bus(:,BUS_TYPE)==PQ);
boundary_bus_ind = setdiff(1:nbus, inner_bus_ind);

Y_red = Y(boundary_bus_ind ,boundary_bus_ind ) ...
    - Y(boundary_bus_ind, inner_bus_ind)*inv(Y(inner_bus_ind,inner_bus_ind))*Y(inner_bus_ind,boundary_bus_ind);

Y_red_real = full( real(Y_red) );
Y_red_imag = full( imag(Y_red) );

write_csv_with_header([note,'_Y_reduced_real_part.csv'],{},Y_red_real);
write_csv_with_header([note,'_Y_reduced_imag_part.csv'],{},Y_red_imag);

% writematrix(Y_red_real,'Y_reduced_real_part.csv');
% writematrix(Y_red_imag,'Y_reduced_imag_part.csv');

%% deal with time series data


% generator data
% 'P_mech(per unit assume constant)'
P_mech_header = {'second'};
for ig = 1:ngen
    P_mech_header = [P_mech_header, ['Pmech-of-gen',num2str(ig)]];
end
write_csv_with_header([note,'_P_mech_in_per_unit.csv'],P_mech_header,[second_series, gen_series']);

% load data
% constant power load for those at generator buses
% load_at_genbus = [mpc.bus(boundary_bus_ind,BUS_I), mpc.bus(boundary_bus_ind,PD)/mpc.baseMVA]';
loadgen_header = {'second'};
for idx = 1:length(boundary_bus_ind)
    bus_num = num2str(mpc.bus(boundary_bus_ind(idx),BUS_I));
    loadgen_header = [loadgen_header, ['load-at-bus-',num2str(bus_num)] ];
end
load_at_genbus = [PD_series(boundary_bus_ind,:)'];
write_csv_with_header([note,'_P_load_at_genbuses.csv'],loadgen_header,[second_series, load_at_genbus]);

% constant current load for those at generator buses
V_load = VM_series(inner_bus_ind, :) .* exp(i*deg2rad(VA_series(inner_bus_ind, :)) );
S_load = PD_series(inner_bus_ind, :) + i*QD_series(inner_bus_ind, :);
I_load = conj( S_load ./ V_load );

I_red = Y(boundary_bus_ind, inner_bus_ind)*inv(Y(inner_bus_ind,inner_bus_ind))*I_load;

% constant current load - magnitudes
I_red_mag_header = {'second','I_red_mag(per unit)'};
write_csv_with_header([note,'_I_red_magnitude.csv'],I_red_mag_header,[second_series, abs(I_red')]);
I_red_ang_header = {'second','I_red_angle(rad)'};
write_csv_with_header([note,'_I_red_angle.csv'],I_red_ang_header,[second_series, angle(I_red')]);

end