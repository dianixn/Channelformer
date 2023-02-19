function local_phi = getLocalPhiBS(theta,phi,alpha,beta,gamma) 
% (M x N) x 1

local_phi = angle(cosd(beta).*sind(theta).*cosd(phi-alpha)-sind(beta).*cosd(theta)+...
    1i.*(cosd(beta).*sind(gamma).*cosd(theta)+(sind(beta).*sind(gamma).*cosd(phi-alpha)+...
    cosd(gamma).*sind(phi-alpha)).*sind(theta))); % radian
local_phi = local_phi.*180./pi; % degree

end
