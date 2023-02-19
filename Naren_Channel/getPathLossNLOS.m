function [PathLossNLOS,ShaNLOS] = getPathLossNLOS(Fc_GHz,Dis3D)
%%% NLOS path model (optional model)
% Input
%       Fc_GHz                 :            carrier frequency in GHz
%       Dis3D                  :            3D BS-UE distance, 1 x the number of NLOS links
% Output
%       PathLossLOS            :            1 x the number of NLOS links
%       ShaNLOS                :            NLOS shadow fading std in dB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PathLossNLOS = 32.4+20*log10(Fc_GHz)+30*log10(Dis3D); 
ShaNLOS = 7.8; % dB
PathLossNLOS = PathLossNLOS+normrnd(0,ShaNLOS,size(Dis3D)); % dB

end