%% description:
% calculation of the Poynting vector of each Fourier harmonic
% inside an inhomogeneous layer
%% input:
% no: number of Fourier harmonics
% V_upward: the vector of size (no, 1) of amplitudes of Fourier harmonics propagating upwards
% V_downward: the vector of size (no, 1) of amplitudes of Fourier harmonics propagating downwards
% kx0: incident plane wave wavevector x-projection (Bloch wavevector)
% kg: wavelength-to-period ratio (grating vector)
% eps1: permittivity of the substrate
% eps2: permittivity of the superstrate
% pol: polarization, either "TE" or "TM"
%% output:
% S_z: the vector of size (no, 1) containing projections of the Poynting vector on the vertical axis 
%% implementation
function [S_z] = calc_Poynting(no, V_upward, V_downward, kx0, kg, eps1, eps2, pol)
    [kz1, kz2] = fmm_kxz(no, kx0, 0, kg, eps1, eps2);
	kz1 = transpose(kz1);
	kz2 = transpose(kz2);
	if (strcmp(pol,'TM'))
		kz1 = kz1/eps1;
		kz2 = kz2/eps2;
    end

    S_z = 0.5 * (abs(V_upward).^2 - abs(V_downward).^2) .* real(kz1) - imag(V_upward .* conj(V_downward)) .* imag(kz1);
end



















