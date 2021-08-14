clear; clc; close all;
define_constants;

%% Case 3: time series from real-world
mpc0 = loadcase('case14.m');

data_folder = 'C:\Users\still\Documents\Github\NSF-1923278-Stochastic-Uncertainty-Management\Data\RawData\DownloadedFromERCOT\';
load_data = readmatrix([data_folder,'load_time_series_1second.csv']);
wind_data = readmatrix([data_folder,'wind_time_series_1second.csv']);

second_series = load_data(:,1);
load_data(:,1) = []; wind_data(:,1) = [];

nt = length(second_series); nbus = size(mpc0.bus,1); ngen = size(mpc0.gen,1);

% mapping regions to 
% bus -- ERCOT Region -- Column in csv
% 3 : COAST : 1
% 14: EAST : 2
% 12: FAR WEST : 3
% 13: NORTH: 4
% 9: NORTH CENTRAL : 5
% 4: SOUTH CENTRAL: 6
% 2: SOUTH : 7
% 5: WEST: 8


power_factor = mpc0.bus(:,QD) ./ mpc0.bus(:,PD);
power_factor(isnan(power_factor)) = 0;
ratio = 5/1000;

load_mapping = [3; 14; 12; 13; 9; 4; 2; 5];
load_series = zeros(nbus,nt);
for il = 1:size(load_data,2)
    load_series(load_mapping(il),:) = load_data(:,il)' * ratio;
end

wind_mapping = [3; 1; 11];
wind_series = zeros(nbus,nt);
for il = 1:size(wind_data,2)
    wind_series(wind_mapping(il),:) = wind_data(:,il)' * ratio;
end

PD_series = load_series - wind_series;
QD_series = PD_series .* repmat(power_factor,1,nt);

PG_series = nan*ones(ngen,nt); VM_series = nan*ones(nbus,nt); VA_series = nan*ones(nbus,nt);
parfor t = 1:1:nt
    if mod(t,1000) ==0
        disp(t/nt);
    end
    mpc_t = mpc0;
    mpc_t.bus(:,PD) = PD_series(:,t);
    mpc_t.bus(:,QD) = QD_series(:,t);
    result_t = runopf(mpc_t, mpoption('verbose',0, 'out.all',0));
    assert( result_t.success );
    PG_series(:,t) = result_t.gen(:,PG);
    VM_series(:,t) = result_t.bus(:,VM);
    VA_series(:,t) = result_t.bus(:,VA);
end

figure;
subplot(3,1,1)
plot(1:nt, PG_series)
xlabel('second')
ylabel('Generation (PG in MW)')
subplot(3,1,2)
plot(1:nt, VM_series)
xlabel('second')
ylabel('Voltage Magnitude (VM in per unit)')
subplot(3,1,3)
plot(1:nt, VA_series)
xlabel('second')
ylabel('Voltage Angle (VA in degree)')

create_network_reduced_files(mpc0, second_series, PG_series/mpc0.baseMVA,...
        PD_series/mpc0.baseMVA, QD_series/mpc0.baseMVA,...
        VM_series, VA_series,'case3_realworld_time_series');
