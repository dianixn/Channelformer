
function local_theta = getLocalThetaBS(theta,phi,alpha,beta,gamma)
% (M x N) x 1

local_theta = acosd(cosd(beta).*cosd(gamma).*cosd(theta)+...
              (sind(beta).*cosd(gamma).*cosd(phi-alpha)-sind(gamma).*sind(phi-alpha)).*sind(theta)); % degree
          
end
