function TransformMatrix = getTransformMatrix(psi)
% Input
%       psi               :   1 x (M x N)
% Output
%       TranformMatrix    :   2 x 2 x (M x N)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

First = cosd(psi); 
Second = sind(psi); 
Third = -Second;
Fourth = First;

TM = [First;Second;Third;Fourth]; % 4 x (M x N)

TransformMatrix = reshape(TM,2,2,length(psi)); % 2 x 2 x (M x N)

end