clc;
clear all;
format long;
seed = 42;
rng(seed); % initialization of random number generator 

%% Parameters of the system
dh_x = 5; % size of cell along x
dh_y = 5; % size of cell along y
dh_z = 5; % size of cell along z
r0_fix = 0.2; % radius of spheres
n0 = 0.1; %packing density of spheres
z_pl = dh_z*rand(1); %coordinate of the secant plane;

[r1, x1, y1, r0, x0, y0, z0] = Spheres_gen(z_pl, dh_x, dh_y, dh_z, r0_fix, n0);


[X,Y,Z] = sphere;

%% Plotting circles
figure();
for i = 1:numel(x1)
    c = [x1(i) y1(i)];
    pos = [c-r1(i) 2*r1(i) 2*r1(i)];
    rectangle('Position',pos,'Curvature',[1 1])
hold on;   
end
plot(-dh_x/2*ones(1, numel(-dh_y/2:dh_y/2)), -dh_y/2:dh_y/2, '--');
plot(dh_x/2*ones(1, numel(-dh_y/2:dh_y/2)), -dh_y/2:dh_y/2,  '--'); 
plot(-dh_x/2:dh_x/2, -dh_y/2*ones(1, numel(-dh_x/2:dh_x/2)), '--');
plot(-dh_x/2:dh_x/2, dh_y/2*ones(1, numel(-dh_x/2:dh_x/2)), '--');
axis equal
xlim ([-dh_x/2-max(r0_fix), dh_x/2+max(r0_fix)])
ylim ([-dh_y/2-max(r0_fix), dh_y/2+max(r0_fix)])

%% Plotting spheres
figure();
axis equal;
hold on;
for i = 1:numel(x0)
    X2 = X*r0(i);
    Y2 = Y*r0(i);
    Z2 = Z*r0(i);
    surf(X2+x0(i),Y2+y0(i),Z2+z0(i), 'LineStyle', 'none', 'FaceColor', 'interp');
end
view(3);