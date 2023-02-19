function AntennaPatternBS = getAntennaPatternBS(theta,phi,orientation,SectorBS,Nt,PolarizationAngle,NumSubPaths,NumCluster)
%%% This function is to calculate BS antenna patterns with polarization (V-POL) but without sectorization. 
%%% Polarization model 2 (Section 7.3.2) is used.  
% Input
%               theta           :       zenith angle of departure (ZOD), (M x N) x 1
%               phi             :       azimuth angle of departure (AOD),(M x N) x 1
%               orientation     :       antenna orientation, 3 x 3
%               SectorBS        :       index of the sector
%               Nt              :       number of transmit antennas
%               PolarizationAngle:       polarization angle, 0 for V-POL
%               NumSubPaths     :       M
%               NumCluster      :       N
% Output
%               AntennaPatternBS :      antenna pattern for the selected sector, 2 x Nt x (M x N)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumAnglesPerPath = NumSubPaths*NumCluster; % MN

orientation = orientation(:,SectorBS);

alpha = orientation(1); 
beta = orientation(2); 
gamma = orientation(3); 

%%% Step 1: Calculate input angles in LCS.

% Obtain angles in LCS.
local_theta = getLocalThetaBS(theta,phi,alpha,beta,gamma);
local_phi = getLocalPhiBS(theta,phi,alpha,beta,gamma);

% Obtain antenna gain in linear unit, MN x 1
AntennaGainLCS = getAntennaGainBS(local_theta,local_phi); 
AntennaGainLCS = sqrt(AntennaGainLCS); 

% Obtain polarization of each antenna
PolPerAntenna = repmat([cosd(PolarizationAngle);sind(PolarizationAngle)],1,Nt); % 2 x Nt

%%% Step 2: Calculate antenna pattern in LCS

FieldPatternLCS = kron(AntennaGainLCS.',PolPerAntenna); % 2 x (Nt x M x N)
FieldPatternLCS = reshape(FieldPatternLCS,2,Nt,NumAnglesPerPath); % 2 x Nt x MN

%%%  Step 3: Transform LCS to GCS

psi = getPsiBS(theta,phi,alpha,beta,gamma); % MN x 1

% Transformation matrix
TransformMatrix = getTransformMatrix(psi.'); % 2 x 2 x MN

% Obtain antenna pattern in GCS
FieldPatternGCS = zeros(size(FieldPatternLCS)); % 2 x Nt x MN

for m = 1:length(psi) % loop over each angle of the link
    
    FieldPatternGCS(:,:,m) = TransformMatrix(:,:,m)*FieldPatternLCS(:,:,m);
    
end

AntennaPatternBS = FieldPatternGCS; % 2 x Nt x MN

end

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

function local_theta = getLocalThetaBS(theta,phi,alpha,beta,gamma)
% (M x N) x 1

local_theta = acosd(cosd(beta).*cosd(gamma).*cosd(theta)+...
              (sind(beta).*cosd(gamma).*cosd(phi-alpha)-sind(gamma).*sind(phi-alpha)).*sind(theta)); % degree
          
end

function local_phi = getLocalPhiBS(theta,phi,alpha,beta,gamma) 
% (M x N) x 1

local_phi = angle(cosd(beta).*sind(theta).*cosd(phi-alpha)-sind(beta).*cosd(theta)+...
    1i.*(cosd(beta).*sind(gamma).*cosd(theta)+(sind(beta).*sind(gamma).*cosd(phi-alpha)+...
    cosd(gamma).*sind(phi-alpha)).*sind(theta))); % radian
local_phi = local_phi.*180./pi; % degree

end

function psi = getPsiBS(theta,phi,alpha,beta,gamma) 
% (M x N) x 1

psi = angle((sind(gamma).*cosd(theta).*sind(phi-alpha)+cosd(gamma).*(cosd(beta).*sind(theta)-sind(beta).*cosd(theta).*cosd(phi-alpha)))...
    +1i.*(sind(gamma).*cosd(phi-alpha)+sind(beta).*cosd(gamma).*sind(phi-alpha))); % radian
psi = psi.*180./pi; % degree

end

function AntennaGainBS = getAntennaGainBS(theta,phi)
% Input
%       theta           :   ZOD in LCS, (M x N) x 1, [0,179] degree
%       phi             :   AOD in LCS, (M x N) x 1, [-179,180] degree
% Output 
%       AntennaGainBS   :   linear unit, (M x N) x 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Table 7.3-1

phi_3dB = 65; 
A_max = 30; % dB
A_H = -min(12*((phi-0)./phi_3dB).^2,A_max);

theta_3dB = 65; 
SLA_V = 30; % dB
A_V = -min(12*((theta-90)./theta_3dB).^2,SLA_V);

gain_max = 8;

A_3D = -min(-(A_V+A_H),A_max);

AntennaGainBS = A_3D+gain_max; % in dBi  
AntennaGainBS = 10.^(AntennaGainBS./10); % linear

end
