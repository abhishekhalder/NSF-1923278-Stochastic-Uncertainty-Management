function mean = weighted_angle_mean(angle,weights)

x = sum(cos(angle).*weights);
y = sum(sin(angle).*weights);

mean = atan2(y,x);