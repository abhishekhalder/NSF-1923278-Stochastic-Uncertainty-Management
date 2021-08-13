clear; clc; close all;

wind_mat = readmatrix('wind_time_series_5min.csv');
wind_mat(:,1) = 0:(5*60):(24*60*60);

load_mat = readmatrix('load_modelA6_time_series_5min.csv');
load_mat(:,1) = 0:(5*60):(24*60*60);

time_1min = 0:1:(24*60*60);

nwind = size(wind_mat,2);
wind_mat_1m = nan*ones(length(time_1min), nwind);
wind_mat_1m(:,1) = time_1min';
for i = 2:nwind
    vq = interp1(wind_mat(:,1), wind_mat(:,i), time_1min, 'makima' );
    wind_mat_1m(:,i) = vq;
end

nload = size(load_mat,2);
load_mat_1m = nan*ones(length(time_1min), nload);
load_mat_1m(:,1) = time_1min';
for i = 2:nload
    vq = interp1(load_mat(:,1), load_mat(:,i), time_1min, 'makima' );
    load_mat_1m(:,i) = vq;
end

fwind = figure('Position',[10,10,500,300]);
plot(time_1min/60/60, wind_mat_1m(:,2:end))
xlabel('Hour')
ylabel('Wind MW')
legend('LZ SOUTH HOUSTON','LZ WEST','LZ NORTH')

fload = figure('Position',[10,10,500,300]);
plot(time_1min/60/60, load_mat_1m(:,2:end))
xlabel('Hour')
ylabel('Load MW')
legend('Coast','East','FarWest','North','NorthCentral','SouthCentral','Southern','West')

writematrix(wind_mat_1m,'wind_time_series_1second.csv')
writematrix(load_mat_1m,'load_time_series_1second.csv')