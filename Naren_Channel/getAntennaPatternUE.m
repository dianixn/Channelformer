function AntennaPatternUE = getAntennaPatternUE(theta,phi,orientation,Nr,PolAngleUE)
%%% This function is to calculate UE antenna patterns with polarization (V-POL).
%%% Polarization model 2 (Section 7.3.2) is used.  
%%% Antenna pattern is calculated for all links of a SINGLE UE. 
%
% Input
%               theta           :       zenith angle of arrival (ZOA), (M x N) x 1
%               phi             :       azimuth angle of arrival (AOA),(M x N) x 1
%               orientation     :       antenna orientation, 3 x 1
%               Nr              :       number of receive antennas
%               PolAngleUE      :       polarization angle, 90
% Output
%               AntennaPatternUE :      antenna pattern for each UE, 2 x Nr x (M x N)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumAngles = length(theta); % (M x N), MN

alpha = orientation(1); 
beta = orientation(2); 
gamma = orientation(3); 

%%% Step 1: Calculate input angles in LCS.

% Isotropic antennas at UE, unit antenna gain, MN x 1
AntennaGainLCS = ones(NumAngles,1); 

% Obtain polarization of each antenna
u = 1:Nr;
PolPerAntenna = [cosd(mod(u,2)*PolAngleUE);sind(mod(u,2)*PolAngleUE)]; % 2 x Nr

%%% Step 2: Calculate antenna pattern in LCS

FieldPatternLCS = kron(AntennaGainLCS.',PolPerAntenna); % 2 x (Nr x MN)
FieldPatternLCS = reshape(FieldPatternLCS,2,Nr,NumAngles); % 2 x Nr x MN

%%%  Step 3: Transform LCS to GCS

psi = getPsiBS(theta,phi,alpha,beta,gamma); % MN x 1

% Transformation matrix
TransformMatrix = getTransformMatrix(psi.');% 2 x 2 x MN

% Obtain antenna pattern in GCS
FieldPatternGCS = zeros(size(FieldPatternLCS));

for m = 1:length(psi) % loop over each angle of each link
    
    FieldPatternGCS(:,:,m) = TransformMatrix(:,:,m)*FieldPatternLCS(:,:,m);
    
end

AntennaPatternUE = FieldPatternGCS; % 2 x Nr x MN

end

function psi = getPsiBS(theta,phi,alpha,beta,gamma) 
% (M x N) x 1

psi = angle((sind(gamma).*cosd(theta).*sind(phi-alpha)+cosd(gamma).*(cosd(beta).*sind(theta)-sind(beta).*cosd(theta).*cosd(phi-alpha)))...
    +1i.*(sind(gamma).*cosd(phi-alpha)+sind(beta).*cosd(gamma).*sind(phi-alpha))); 
psi = psi.*180./pi; % degree

end




