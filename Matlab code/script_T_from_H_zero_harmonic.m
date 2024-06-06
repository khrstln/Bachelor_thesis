%% script to calculate the dependence of the reflected (R_sum) and transmitted (T_sum) energy on the thickness of the sample 

clc;
clear all;
format long;


seed = 0;
rng(seed); % initialization of random number generator 



%% Parameters of the system
wl = 0.5; %wavelength in micrometers
dl = 15; % period of the structure in micrometers 2
dh = 50; % layer thickness in micrometers 
epss = 4; % dielectric permittivity of cylinders 
epsm = 1; %  dielectric permittivity of background media
r0_fix = 0.8; % radius of cylinders
n0 = 0.5; %packing density of cylinders



n_var = 250; % ensemble size for averaging



dn_sl = 41; %number of slices per 1 cylinder
n_cyl = floor(dh*dl/pi./r0_fix^2*n0); % number of cylinders in the unit cell
n_cyl_tot = n_cyl; %total number of cylinders; for const radius n_cyl_tot = n_cyl
n_sl = dn_sl*ceil(dh/(2*max(r0_fix))); % total number of slices per unit cell
dy = dh/n_sl; % slice thickness
delta = 2*sqrt(r0_fix^2 - (r0_fix-2*dy)^2); %the smallest feature
% no = 2*ceil(max(4*dl/wl, 2*dl/delta)); % number of Fourier modesunder consideration
min_dist = dl/1000; % min distance 
no=150;
%% Target values
grid = 1:10:n_sl; %grid of slices for measurements
writematrix(grid*dy, strcat("C:\Users\iliya.hrustalev\Matlab code\different n0\T(H) n0=", num2str(n0), "\r0=", num2str(r0_fix), "\grid.txt"))

R_sum = zeros(numel(grid), n_var); %array of total reflected energy
T_sum = zeros(numel(grid), n_var); %array of total reflected energy

%% ensamble averaging

for i_var = 1:n_var
    disp([num2str(i_var), ' iter of ', num2str(n_var)])
%% Generation of spheres
    x0 = ones(1, 2*n_cyl_tot)*2*max(dh, dl); 
    y0 = ones(1, 2*n_cyl_tot)*2*max(dh, dl); 
    r0 = ones(1, 2*n_cyl_tot)*min(r0_fix);

    i_cyl = zeros(1, numel(n_cyl)); % counter of real cylinders
    i_cyl_tot = zeros(1, numel(n_cyl)); % counter of real + mirrored cylinders


    while i_cyl < n_cyl        
        % new coordinate generation
        x = -dl/2 + dl * rand(1);
        i_r = randi([1,numel(r0_fix)]); % i_r = 1 for constant radius
        r = r0_fix(i_r); 
        
        
        inbound = ( x < (-dl/2 + r)) | ( x > (dl/2 - r)); %intersections with boundaries
        y = r + (dh-2*r) * rand(1);
        inters = 0;

        %cylihnders intersection check
        if i_cyl >= 0 
          dr = ((x-x0).^2 + (y-y0).^2) .^ (0.5) - (r+r0) - min_dist; % distances to other cylinders
          drm = dr; % distances to other cylinders from mirrored coordinate
          if inbound
              xm = ( x < (-dl/2 + r))*(dl+x) + ( x > (dl/2 - r))*(-dl+x); % mirrored coordinate
              drm = ((xm-x0).^2 + (y-y0).^2) .^ (0.5) - (r+r0) - min_dist;
          end
          inters = numel([dr(dr<0) drm(drm<0)]);
        end



        %adding a new cylinder to arrays
        if inters == 0
            i_cyl(i_r) = i_cyl(i_r) + 1;
            i_cyl_tot(i_r) = i_cyl_tot(i_r) + 1;
            x0(i_cyl_tot(i_r)) = x;
            y0(i_cyl_tot(i_r)) = y;
            r0(i_cyl_tot(i_r)) = r;
            if inbound
                i_cyl_tot(i_r) = i_cyl_tot(i_r) + 1;
                x0(i_cyl_tot(i_r)) = xm;
                y0(i_cyl_tot(i_r)) = y;
                r0(i_cyl_tot(i_r)) = r;
            end
        end   

    end
    x0 = x0(x0 < 2*max(dh, dl));
    y0 = y0(y0 < 2*max(dh, dl));
    r0 = r0(1:numel(x0));

%% Plotting cylinders
    % figure();
    %  for i = 1:i_cyl_tot
    %      c = [x0(i) y0(i)];
    %      pos = [c-r0(i) 2*r0(i) 2*r0(i)];
    %      rectangle('Position',pos,'Curvature',[1 1])
    %      hold on;   
    %  end
    % plot(-dl/2*ones(1, numel(0:dh)), 0:dh, '--');
    %  plot(dl/2*ones(1, numel(0:dh)), 0:dh,  '--');
    %  axis equal
    % xlim ([-dl/2-max(r0_fix), dl/2+max(r0_fix)])
    % ylim ([0, dh])


%% Slicing
%% plotting slices
        % for i = 0:(n_sl-1)
        %     plot(-dl/2:dl/2, i*dy*ones(1, numel(-dl/2:dl/2)), 'Color', 'black')
        % end

   
    c_x = 100*dl*ones(1, n_sl * ceil(dl/(min(r0_fix)))*4); % ractangle center coordinate
    w_r = 100*dl*ones(1, n_sl * ceil(dl/(min(r0_fix)))*4); % rectagle width
    eps_r = zeros(1, n_sl * ceil(dl/(min(r0_fix)))*4); % array of permittivities 
    n_mesh = zeros(1, n_sl); % number of rectangles in each slice
    
    i_rect = 0; % number of rectangles
    for i_sl = 1:n_sl  
            ymin = (i_sl-1)*dy; % lower boundary of the slice
            ymax = i_sl*dy;  % upper boundary of the slice

            %cylinders that intersect this slice
            dy_min = dy/3; % min cylinder-slice intersection
            x0_in = x0( (y0+r0 > ymin+dy_min) & (y0-r0 < ymax-dy_min) );
            y0_in = y0( (y0+r0 > ymin+dy_min) & (y0-r0 < ymax-dy_min) );
            r_in = r0( (y0+r0 > ymin+dy_min) & (y0-r0 < ymax-dy_min) );


            [x0_sort, I] = sort(x0_in);
            y0_sort = y0_in(I);
            r_sort = r_in(I);


            i_rect_old = i_rect;
            for i_cyl = 1:numel(x0_sort)

                rs = r_sort(i_cyl);
                ys = y0_sort(i_cyl);
                xs = x0_sort(i_cyl);

                %calculating boundaries of rect:
                l1 = min(rs, abs(ymax - ys));
                l2 = min(rs, abs(ys - ymin));
                ws = 2*max((rs^2 - l1^2)^(0.5), (rs^2 - l2^2)^(0.5)); 
                xl = max(-dl/2, xs - ws/2); % left edge
                xr = min(dl/2, xs + ws/2);  % right edge

                min_dist = dl*10^(-12);
                rect_int = 0;
                if xr - xl > min_dist 
                    x_c = (xl+xr)/2; % center
                    r_w = (xr-xl); % width      

                    if i_rect-i_rect_old == 0 %? first cylinder
                        if (xl -(-dl/2)) > min_dist
                            %adding first rect with eps = epsm
                            i_rect = i_rect+1;
                            w_r(i_rect) = (xl - (-dl/2));
                            c_x(i_rect) = xl - w_r(i_rect)/2;                
                            eps_r(i_rect) = epsm;

                        end
                    elseif (xl - (c_x(i_rect) + w_r(i_rect)/2))>0
                         %adding intermediate rect with eps = epsm
                         i_rect = i_rect+1;
                         w_r(i_rect) = (xl - (c_x(i_rect-1) + w_r(i_rect-1)/2));
                         c_x(i_rect) = (xl + (c_x(i_rect-1) + w_r(i_rect-1)/2))/2;
                         eps_r(i_rect) = epsm;  


                    else % if rectangles for 2 different cylinders  intersect
                       w_r_old = w_r(i_rect);
                       w_r(i_rect) = xr -  (c_x(i_rect) - w_r(i_rect)/2);
                       c_x(i_rect) = (xr +  (c_x(i_rect) - w_r_old/2))/2;
                       rect_int = 1;
                    end


                    if rect_int == 0
                    %adding rect with eps = epss    
                        i_rect = i_rect+1;            
                        c_x(i_rect) = x_c;
                        w_r(i_rect) = r_w;
                        eps_r(i_rect) = epss;    
                    end

                end       

            end
            %adding last rect with eps = epsm
            if (i_rect - i_rect_old) == 0 % layer is empty
                i_rect = i_rect+1;  
                c_x(i_rect) = 0;
                w_r(i_rect) = dl;
                eps_r(i_rect) = epsm;
            elseif c_x(i_rect) + w_r(i_rect)/2 + min_dist < dl/2
               i_rect = i_rect+1;            
               c_x(i_rect) = (dl/2 + c_x(i_rect-1) + w_r(i_rect-1)/2)/2;
               a = c_x(i_rect);
               w_r(i_rect) = (dl/2 - (c_x(i_rect-1) + w_r(i_rect-1)/2));
               eps_r(i_rect) = epsm;   
            end
            n_mesh(i_sl) = i_rect-i_rect_old; % save number of rectangles in slice 

     end

        c_x = c_x(c_x<100*dl)/dl;
        w_r = w_r(w_r<100*dl)/dl;
        eps_r = eps_r(eps_r ~= 0);     


%% Plotting rectangles 
        % for i_sl = 1:numel(n_mesh)
        %     n_rec = n_mesh(i_sl);
        %     y = (i_sl - 1) * dy/dl;
        %     h = dy/dl;
        %     for i_rec = 1:n_rec
        %       i_x = sum(n_mesh(1:(i_sl-1))) + i_rec;
        %       x = c_x(i_x) - w_r(i_x)/2;
        %       w = w_r(i_x);
        %       if eps_r(i_x) == epsm
        %         rectangle('Position', [x y w h]*dl, 'EdgeColor', 'k', 'FaceColor', 'b');
        %       else
        %           rectangle('Position', [x y w h]*dl, 'EdgeColor', 'k');
        %       end
        %     end
        % end
        % hold off;

        
        
        c_x = c_x(numel(c_x):-1:1);
        w_r = w_r(numel(w_r):-1:1);
        eps_r = eps_r(numel(eps_r):-1:1);
        n_mesh = n_mesh(numel(n_mesh):-1:1);
        

        
%}
%% Fourier Modal Method
        %Paramaters of inc wave
        wv = 2*pi/wl; % wavevector
        pol = 'TE'; % polarization, "TE" or "TM"
        %grating parameters
        gp = dl; % grating period
        gh = dy; % grating depth
        %permittivities
        eps_sub = epsm; % substrate permittivity
        eps_sup = epsm; % superstrate permittivity
        %method parameters        
        ind0 = ceil(no/2); % index of the zero harmonic (0th order diffraction)
        %incidence
        theta = 0.00001; % angle of incidence
        kx0 = sin(theta*pi/180); % incidence wavevector projection
        
        V_inc = zeros(no,2); % matrix of incident field amplitudes
        V_inc(ind0,2) = 1; % plane wave coming from the superstrate

        i_grid = 1;
        
        n_mesh_inv = n_mesh(end:-1:1);
        w_r_inv = w_r(end:-1:1);
        c_x_inv = c_x(end:-1:1);
        eps_r_inv = eps_r(end:-1:1);
        SM_down = 0;
        SM_down_arr = zeros(no, no, 2, 2, length(grid));
        for i_sl = 1:n_sl
            % Slice parameters
            n_rec = n_mesh_inv(i_sl);  % number of rectangles in this slice  
            i0 = sum(n_mesh_inv(1:(i_sl-1)))+1; % position 
            alps = w_r_inv(i0:(i0+n_rec-1));
            poss = c_x_inv(i0:(i0+n_rec-1));
            eps = eps_r_inv(i0:(i0+n_rec-1));
            % calculate Fourier image matrix of the dielectric permittivity function
            FM = calc_emn_bin(no, alps, poss, eps);
            % scattering matrix of the grating of size (no,no,2,2)
            % block SM(:,:,1,1) corresponds to refelection from substrate to substrate
            % block SM(:,:,2,2) corresponds to refelection from superstrate to superstrate
            % block SM(:,:,2,1) corresponds to transmission from substrate to superstrate
            % block SM(:,:,1,2) corresponds to transmission from superstrate to substrate
            SM = fmm(no, kx0, wl/gp, wv*gh, eps_sub, eps_sup, FM, pol);

            if numel(SM_down) ~= 1 %? first slice in a layer               
                SM_down = mul_SM(SM_down, SM);
            else 
                SM_down = SM;
            end
            
            if i_sl == grid(min(i_grid, numel(grid)))
                SM_down_arr(:, :, :, :, i_grid) = SM_down;
                i_grid = i_grid + 1;
            end

        end

        i_grid = 1;
        SM_full_slab = zeros(2*no,2*no);
        SM_up = 0;
        for i_sl = 1:n_sl
            %Slice parameters
            n_rec = n_mesh(i_sl);  % number of rectangles in this slice  
            i0 = sum(n_mesh(1:(i_sl-1)))+1; % position 
            alps = w_r(i0:(i0+n_rec-1));
            poss = c_x(i0:(i0+n_rec-1));
            eps = eps_r(i0:(i0+n_rec-1));
            % calculate Fourier image matrix of the dielectric permittivity function
            FM = calc_emn_bin(no, alps, poss, eps);
            % scattering matrix of the grating of size (no,no,2,2)
            % block SM(:,:,1,1) corresponds to refelection from substrate to substrate
            % block SM(:,:,2,2) corresponds to refelection from superstrate to superstrate
            % block SM(:,:,2,1) corresponds to transmission from substrate to superstrate
            % block SM(:,:,1,2) corresponds to transmission from superstrate to substrate
            SM = fmm(no, kx0, wl/gp, wv*gh, eps_sub, eps_sup, FM, pol);

            if numel(SM_up) ~= 1 %? first slice in a layer               
                SM_up = mul_SM(SM, SM_up);
            else 
                SM_up = SM;
            end    

            
            if i_sl == grid(min(i_grid, numel(grid)))
                i = max(1, length(grid) - i_grid);
                SM_down = SM_down_arr(:,:,:,:,i);
                
                SM_mult = mul_SM(SM_down, SM_up);
                SM_full_slab(1:no,1:no) = SM_mult(:,:,1,1);
                SM_full_slab(no+1:2*no,1:no) = SM_mult(:,:,2,1); 
                SM_full_slab(1:no,no+1:2*no) = SM_mult(:,:,1,2); 
                SM_full_slab(no+1:2*no,no+1:2*no) = SM_mult(:,:,2,2);
                % 
                % writematrix(SM_full_slab, "C:\Users\HP\Desktop\code\S-matrix full slab\SM_full_slab_" + num2str(i_grid) + ".csv");

                A_1 = eye(no) - SM_down(:,:,2,2) * SM_up(:,:,1,1);
                b_1 = SM_down(:,:,2,2) * SM_up(:,:,1,2) * V_inc(:,2) + SM_down(:,:,2,1) * V_inc(:,1);
                V_upward = linsolve(A_1, b_1);

                A_2 = eye(no) - SM_up(:,:,1,1) * SM_down(:,:,2,2);
                b_2 = SM_up(:,:,1,2) * V_inc(:,2) + SM_up(:,:,1,1) * SM_down(:,:,2,1) * V_inc(:,1);
                V_downward = linsolve(A_2, b_2);

                
                Poyting_z = calc_Poynting(no, V_upward, V_downward, kx0, wl/gp, eps_sub, eps_sup, pol);
                
            	[kz1, kz2] = fmm_kxz(no, kx0, 0, wl/gp, eps_sub, eps_sup);
                kz1 = transpose(kz1);
                kz2 = transpose(kz2);

                V_dif = zeros(no, 2);

                V_dif(:,2) = SM_mult(:,:,2,2) * V_inc(:,2);

                Poynting_reflected = fmm_efficiency(no, V_inc, V_dif, kx0, wl/gp, eps_sub, eps_sup, pol);

                P_inc = 0.5*sum( abs(V_inc(:,2).^2).*real(kz2) );
                % R_sum(i_grid, i_var) = sum(Poynting_reflected(:,2)); % transmitted energy
%                 T_sum(i_grid, i_var) = sum(Poyting_z/P_inc); % transmitted energy     
                T_sum(i_grid, i_var) = (Poyting_z(ind0)/P_inc); % transmitted energy zero harmonic
                i_grid  = i_grid + 1;
            end
        end
       
writematrix(T_sum(:,i_var), strcat("C:\Users\iliya.hrustalev\Matlab code\different n0\T(H) n0=", num2str(n0), "\r0=", num2str(r0_fix), "\T_", num2str(i_var), ".txt"))       
        
   
end
%% plotting obtained dependecy

% R_av = mean(R_sum, 2);
T_av = mean(T_sum, 2);
writematrix(T_av, strcat("C:\Users\iliya.hrustalev\Matlab code\different n0\T(H) n0=", num2str(n0), "\r0=", num2str(r0_fix), "\T_av.txt"))
% set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
% set(groot, 'defaultLegendInterpreter','latex');
% 
% % csvwrite('T_1.csv',T_av);
% % csvwrite('grid_1.csv',grid);
% 
% figure()
% hold on;
% ax = gca;
% ax.FontSize = 14;
% % plot(grid*dy, R_av, 'LineWidth', 2, 'Color', 'r');
% plot(grid*dy, T_av, 'LineWidth', 2, 'Color', 'b');
% % legend({"R","T"})
% xlabel('Depth, H', 'fontsize', 18, 'Interpreter', 'Latex');
% ylabel('T(H) averaged', 'fontsize', 18, 'Interpreter', 'Latex');
