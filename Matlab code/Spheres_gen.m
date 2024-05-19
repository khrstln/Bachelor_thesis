%% input:
% z_pl: coordinate of the secant plane;
% dh_x: size of cell along x
% dh_y: size of cell along y
% dh_z: size of cell along z
% r0_fix: radius of spheres
% n0: packing density of sphere
%% output:
% r1: radii of the spheres intersecting with the plane
% x1: x-coordinates of the centers of the spheres intersecting with the plane
% y1: y-coordinates of the centers of the spheres intersecting with the plane
% r0: radii of all spheres
% x0: x-coordinates of the centers of all spheres
% y0: y-coordinates of the centers of all spheres
% z0: z-coordinates of the centers of all spheres

function [r1, x1, y1, r0, x0, y0, z0] = Spheres_gen(z_pl, dh_x, dh_y, dh_z, r0_fix, n0)

    n_spheres = floor((dh_x*dh_y*dh_z)/(4*pi*r0_fix^3/3)*n0); % number of spheres in the unit cell
    n_spheres_tot = n_spheres; %total number of spheres; for const radius n_cyl_tot = n_cyl
    min_dist = min([dh_x,dh_y,dh_z])/500; % min distance
    
    x0 = ones(1, 2*n_spheres_tot)*2*max([dh_x, dh_y, dh_z]); 
    y0 = ones(1, 2*n_spheres_tot)*2*max([dh_x, dh_y, dh_z]);
    z0 = ones(1, 2*n_spheres_tot)*2*max([dh_x, dh_y, dh_z]);
    r0 = ones(1, 2*n_spheres_tot)*min(r0_fix);
    
    i_spheres = zeros(1, numel(n_spheres)); % counter of real spheres
    i_spheres_tot = zeros(1, numel(n_spheres)); % counter of real + mirrored spheres
    
    while i_spheres < n_spheres
        % new coordinate generation
        x = -dh_x/2 + dh_x * rand(1);
        y = -dh_y/2 + dh_y * rand(1);
        z = -dh_z/2 + dh_z * rand(1);
        i_r = randi([1,numel(r0_fix)]); % i_r = 1 for constant radius
        r = r0_fix(i_r); 
        
        inbound_x = ( x < (-dh_x/2 + r)) | ( x > (dh_x/2 - r)); %intersections with x-boundaries
        inbound_y = ( y < (-dh_y/2 + r)) | ( y > (dh_y/2 - r)); %intersections with y-boundaries
        z = r + (dh_z-2*r) * rand(1);
        inters = 0;
    
        %spheres intersection check
        if i_spheres >= 0 
          dr = ((x-x0).^2 + (y-y0).^2 + (z-z0).^2) .^ (0.5) - (r+r0) - min_dist; % distances to other spheres
          drm = dr; % distances to other spheres from mirrored coordinate
          if inbound_x && inbound_y
              xm = ( x < (-dh_x/2 + r))*(dh_x+x) + ( x > (dh_x/2 - r))*(-dh_x+x); % mirrored coordinate
              ym = ( y < (-dh_y/2 + r))*(dh_y+y) + ( y > (dh_y/2 - r))*(-dh_y+y); % mirrored coordinate
              drm = ((xm-x0).^2 + (ym-y0).^2 + (z-z0).^2) .^ (0.5) - (r+r0) - min_dist;
          elseif not(inbound_x) && inbound_y
              ym = ( y < (-dh_y/2 + r))*(dh_y+y) + ( y > (dh_y/2 - r))*(-dh_y+y); % mirrored coordinate
              drm = ((x-x0).^2 + (ym-y0).^2 + (z-z0).^2) .^ (0.5) - (r+r0) - min_dist;
          elseif inbound_x && not(inbound_y)
              xm = ( x < (-dh_x/2 + r))*(dh_x+x) + ( x > (dh_x/2 - r))*(-dh_x+x); % mirrored coordinate
              drm = ((xm-x0).^2 + (y-y0).^2 + (z-z0).^2) .^ (0.5) - (r+r0) - min_dist;
          end
          inters = numel([dr(dr<0) drm(drm<0)]);
        end
    
    
    
        %adding a new spheres to arrays
        if inters == 0
            i_spheres(i_r) = i_spheres(i_r) + 1;
            i_spheres_tot(i_r) = i_spheres_tot(i_r) + 1;
            x0(i_spheres_tot(i_r)) = x;
            y0(i_spheres_tot(i_r)) = y;
            z0(i_spheres_tot(i_r)) = z;
            r0(i_spheres_tot(i_r)) = r;
            if inbound_x && inbound_y
                i_spheres_tot(i_r) = i_spheres_tot(i_r) + 1;
                x0(i_spheres_tot(i_r)) = xm;
                y0(i_spheres_tot(i_r)) = ym;
                z0(i_spheres_tot(i_r)) = z;
                r0(i_spheres_tot(i_r)) = r;
            elseif not(inbound_x) && inbound_y
                i_spheres_tot(i_r) = i_spheres_tot(i_r) + 1;
                x0(i_spheres_tot(i_r)) = x;
                y0(i_spheres_tot(i_r)) = ym;
                z0(i_spheres_tot(i_r)) = z;
                r0(i_spheres_tot(i_r)) = r;
            elseif inbound_x && not(inbound_y)
                i_spheres_tot(i_r) = i_spheres_tot(i_r) + 1;
                x0(i_spheres_tot(i_r)) = xm;
                y0(i_spheres_tot(i_r)) = y;
                z0(i_spheres_tot(i_r)) = z;
                r0(i_spheres_tot(i_r)) = r;
            end
        end
 
    end
    x0 = x0(x0 < 2*max([dh_x, dh_y, dh_z]));
    y0 = y0(y0 < 2*max([dh_x, dh_y, dh_z]));
    z0 = z0(z0 < 2*max([dh_x, dh_y, dh_z]));
    r0 = r0(1:numel(x0));
    
    x1 = [];
    y1 = [];
    z1 = z0;
    r1 = [];
    
    for i_z = 1:numel(z0)
        if abs(z_pl - z0(i_z)) < r0(i_z)
            x1 = [x1 x0(i_z)];
            y1 = [y1 y0(i_z)];
            r1 = [r1 (r0(i_z)^2 - (z_pl - z0(i_z))^2)^0.5];
        end
    end
end



