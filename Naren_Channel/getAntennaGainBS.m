
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
