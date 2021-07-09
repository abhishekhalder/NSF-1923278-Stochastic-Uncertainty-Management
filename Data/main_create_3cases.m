clear; clc; close all;
define_constants;
%% Main function
% create_network_reduced_files(mpc, second_series, gen_series, PD_series, QD_series,...
%                            VM_series, VA_series, note)

%% Case 0: 
mpc0 = loadcase('case14.m');
result0 = runopf(mpc0);
create_network_reduced_files(result0, 0, result0.gen(:,PG)/result0.baseMVA,...
        result0.bus(:,PD)/result0.baseMVA, result0.bus(:,QD)/result0.baseMVA,...
        result0.bus(:,VM), result0.bus(:,VA),'case0_norminal');


%% Case 1: contingency
% line 13 failure
failed_line = 13;
mpc1 = result0;
mpc1.branch(failed_line, BR_STATUS) = 0;
create_network_reduced_files(mpc1, 0, result0.gen(:,PG)/result0.baseMVA,...
        result0.bus(:,PD)/result0.baseMVA, result0.bus(:,QD)/result0.baseMVA,...
        result0.bus(:,VM), result0.bus(:,VA),['case1_line_',num2str(failed_line),'_failure']);

%% Case 2: time series
% a time series
mpc2 = result0;
% data is from July.08.2012, 12:00 to 23:59, resolution is 30min
% load_data = readcsv('.\IEEE-14-Bus-System\LoadMaxPower_0708_2012.csv');
nt = 24; nbus = size(mpc2.bus,1); ngen = size(mpc2.gen,1);
load_scaling_series = 0.5+sin(pi*(0:1/(nt-1):1))/2; % a toy case 
assert( size(load_scaling_series,2)==nt );
PD_series = repmat(load_scaling_series, nbus, 1) .* mpc2.bus(:,PD);
QD_series = repmat(load_scaling_series, nbus, 1) .* mpc2.bus(:,QD);

PG_series = zeros(ngen,nt); VM_series = zeros(nbus,nt); VA_series = zeros(nbus,nt);
for t = 1:nt
    mpc_t = mpc2;
    mpc_t.bus(:,PD) = PD_series(:,t);
    mpc_t.bus(:,QD) = QD_series(:,t);
    result_t = runopf(mpc_t);
    assert( result_t.success );
    PG_series(:,t) = result_t.gen(:,PG);
    VM_series(:,t) = result_t.bus(:,VM);
    VA_series(:,t) = result_t.bus(:,VA);
end

figure;
subplot(3,1,1)
plot(1:nt, PG_series)
subplot(3,1,2)
plot(1:nt, VM_series)
subplot(3,1,3)
plot(1:nt, VA_series)

second_series = 1800*(0:1:(nt-1))';
create_network_reduced_files(mpc2, second_series, PG_series/mpc2.baseMVA,...
        PD_series/mpc2.baseMVA, QD_series/mpc2.baseMVA,...
        VM_series, VA_series,'case2_time_series');

    
% PD_series = zeros(nbus,nt);
% load_buses = find(mpc2.bus(:,PD) > 0.1);
% assert(length(load_buses)==size(load_data,2));
% PD_series(load_buses,:) = load_data';
% power_factor = mpc2.bus(:,QD) ./ mpc2.bus(:,PD);
% power_factor(isnan(power_factor)) = 0;
% QD_series = PD_series .* repmat(power_factor, nt);