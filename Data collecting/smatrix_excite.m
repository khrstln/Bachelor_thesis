%%
	% (+below,-above) -> (+,-)
%%
function [Vexc] = smatrix_excite(S1, S2, Vinc)
	no = size(S1,1);
	Vexc = Vinc;
	if (numel(size(S1)) == 4) && (numel(size(S2)) == 4)
        Vexc(:,1) = (eye(no) - S1(:,:,2,2)*S2(:,:,1,1)) \ ...
                ( S1(:,:,2,1)*Vinc(:,1) + S1(:,:,2,2)*S2(:,:,1,2)*Vinc(:,2) );
		Vexc(:,2) = (eye(no) - S2(:,:,1,1)*S1(:,:,2,2)) \ ...
                ( S2(:,:,1,1)*S1(:,:,2,1)*Vinc(:,1) + S2(:,:,1,2)*Vinc(:,2) );
	elseif (numel(size(S1)) == 3) && (numel(size(S2)) == 4)
		Vexc(:,1) = (eye(no) - S1(:,2,2).*S2(:,:,1,1)) \ ...
								( S1(:,2,1).*Vinc(:,1) + S1(:,2,2).*(S2(:,:,1,2)*Vinc(:,2)) );
		Vexc(:,2) = (eye(no) - S2(:,:,1,1).*transpose(S1(:,2,2))) \ ...
								( S2(:,:,1,1)*(S1(:,2,1).*Vinc(:,1)) + S2(:,:,1,2)*Vinc(:,2) );
	elseif (numel(size(S1)) == 4) && (numel(size(S2)) == 3)
		Vexc(:,1) = (eye(no) - S1(:,:,2,2).*transpose(S2(:,1,1))) \ ...
								( S1(:,:,2,1)*Vinc(:,1) + S1(:,:,2,2)*(S2(:,1,2).*Vinc(:,2)) );
		Vexc(:,2) = (eye(no) - S2(:,1,1).*S1(:,:,2,2)) \ ...
								( S2(:,1,1).*(S1(:,:,2,1)*Vinc(:,1)) + S2(:,1,2).*Vinc(:,2) );
	elseif (numel(size(S1)) == 3) && (numel(size(S2)) == 3)
		Vexc(:,1) = ( S1(:,2,1).*Vinc(:,1) + S1(:,2,2).*S2(:,1,2).*Vinc(:,2) ) ./ (1 - S1(:,2,2).*S2(:,1,1));
		Vexc(:,2) = ( S2(:,1,1).*S1(:,2,1).*Vinc(:,1) + S2(:,1,2).*Vinc(:,2) ) ./ (1 - S2(:,1,1).*S1(:,2,2));
	else
		error('incorrect input size');
	end
end