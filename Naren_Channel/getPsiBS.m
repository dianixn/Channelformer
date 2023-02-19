
function psi = getPsiBS(theta,phi,alpha,beta,gamma) 
% (M x N) x 1

psi = angle((sind(gamma).*cosd(theta).*sind(phi-alpha)+cosd(gamma).*(cosd(beta).*sind(theta)-sind(beta).*cosd(theta).*cosd(phi-alpha)))...
    +1i.*(sind(gamma).*cosd(phi-alpha)+sind(beta).*cosd(gamma).*sind(phi-alpha))); % radian
psi = psi.*180./pi; % degree

end
