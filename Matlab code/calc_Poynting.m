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



















