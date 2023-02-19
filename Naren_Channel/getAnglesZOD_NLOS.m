function [ClusterZOD,RayZOD] = getAnglesZOD_NLOS(PowersNLOS,NumClusterNLOS,ZSD,ZOD,RayOffset,C_Ele_NLOS,mu_zsd,Fc_GHz,Dis2D_NLOS,UE_heightNLOS)
%%% This function is to calculate 
%                                 1. cluster zenith angle of departure (ZOD) of L NLOS links; 
%                                 2. subpath zenith angle of departure (ZOD) of L NLOS links. 
% Input 
%       PowersNLOS        :   cluster power, N x L
%       NumClusterNLOS    :   number of clusters, N
%       ZSD               :   azimuth spread of arrival, 1 x L
%       ZOD               :   LOS azimuth angle of arrival of direct paths, 1 x L
%       RayOffset         :   subpath angle offset, M x 1
%       C_Ele_NLOS        :   scaling factor, 1 x L
%       mu_zsd            :   1 x L
%       Fc_GHz            :   carrier frequency in GHz
%       Dis2D_NLOS        :   L x 1
%       UE_heightNLOS     :   L x 1
% Output
%       ClusterZOD       :   cluster ZOD, N x L
%       RayZOD           :   subpath ZOD, (M x N) x L
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(RayOffset); % M
NumNLOS = length(ZSD); % L

% Adjust vector dimensions, N x L.
ZSD = repmat(ZSD,NumClusterNLOS,1); 
ZOD = repmat(ZOD,NumClusterNLOS,1); 
mu_zsd = repmat(mu_zsd,NumClusterNLOS,1); 
C_Ele_NLOS = repmat(C_Ele_NLOS,NumClusterNLOS,1); 

% Obtain cluster ZOD, N x L.
ClusteZOD_temp = getClusterZOD_NLOS(ZSD,ZOD,PowersNLOS,C_Ele_NLOS,NumClusterNLOS,Fc_GHz,Dis2D_NLOS,UE_heightNLOS); 

% Obtain subpath ZOD, M x (N x L).
RayZOD_temp = getRayZOD_NLOS(ClusteZOD_temp,RayOffset,mu_zsd); 

% Obtain cluster ZOD within [0 179] degree.
ClusterZOD = wrapElevation(ClusteZOD_temp); 

% Randomly coupling subpaths within each cluster, M x (N x L)
RayZOD = Coupling(RayZOD_temp,M,NumClusterNLOS*NumNLOS);

% (M x N) x L
RayZOD = reshape(RayZOD,M*NumClusterNLOS,NumNLOS); 

end

function ClusterZOD = getClusterZOD_NLOS(ZSD,ZOD,PowersNLOS,C_Ele_NLOS,NumClusterNLOS,Fc_GHz,Dis2D_NLOS,UE_heightNLOS)
%%% This function is to calculate cluster zenith angle of departure (ZOD) of L NLOS links; 
%
% Input 
%       ZSD               :   azimuth spread of arrival, 1 x L
%       ZOD               :   LOS azimuth angle of arrival of direct paths, 1 x L
%       PowersNLOS        :   cluster power, N x L
%       C_Ele_NLOS        :   scaling factor, 1 x L
%       NumClusterNLOS    :   number of clusters, N
%       Fc_GHz            :   carrier frequency in GHz
%       Dis2D_NLOS        :   L x 1
%       UE_heightNLOS     :   L x 1
% Output
%       ClusterZOD       :   cluster ZOD, N x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute ZOD offset (Table 7.5-7).
a = 0.208*log10(Fc_GHz)-0.782;
b = 25;
c = -0.13*log10(Fc_GHz)+2.03;
e = 7.66*log10(Fc_GHz)-5.96;
mu_offset = e-10.^(a*log10(max(b,Dis2D_NLOS))+c-0.07*(UE_heightNLOS-1.5)); % L x 1
mu_offset = repmat(mu_offset.',NumClusterNLOS,1); % N x L

% N x L
ZOD_prime = -ZSD.*log(PowersNLOS./repmat(max(PowersNLOS,[],1),NumClusterNLOS,1))./C_Ele_NLOS;
X = sign(rand(size(PowersNLOS))-0.5);
Y = normrnd(0,ZSD./7,size(ZSD));

ClusterZOD = X.*ZOD_prime+Y+ZOD+mu_offset;

end
function RayZOD = getRayZOD_NLOS(ClusteZOD_temp,RayOffset,mu_zsd)
%%% This function is to calculate subpath zenith angle of departure (ZOD) of L NLOS links. 
% 
% Input 
%       ClusteZOD_temp           :   cluster ZOD,  N x L
%       RayOffset                :   subpath angle offset, M x 1
%       mu_zsd                   :   1 x L

% Output
%       RayZOD                   :   subpath ZOD, M x (N x L)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(RayOffset);
NumCluster = size(ClusteZOD_temp,1);
NumNLOS = size(ClusteZOD_temp,2);

ClusteZOD_temp = reshape(ClusteZOD_temp,1,NumCluster*NumNLOS); % 1 x (N x L)
ClusteZOD_temp = repmat(ClusteZOD_temp,M,1); % M x (N x L)
mu_zsd = reshape(mu_zsd,1,NumCluster*NumNLOS); % 1 x (N x L)
mu_zsd = repmat(mu_zsd,M,1); % M x (N x L)

RayZOD_temp = ClusteZOD_temp+(3/8)*(10.^mu_zsd).*repmat(RayOffset,1,NumCluster*NumNLOS);

RayZOD = wrapElevation(RayZOD_temp);

end



function Angles_ray = Coupling(Angles_ray_temp,M,N)
%%% Coupling subpaths within one cluster

[~,order] = sort(rand(M,N),1);
index = order+repmat([1:M:M*N],M,1)-1;
Angles_ray_temp = Angles_ray_temp(index);
Angles_ray = reshape(Angles_ray_temp,M,N);

end


function ElevationAngles = wrapElevation(angles)
%%% [0,179]

angles(logical(angles<0)) = -angles(logical(angles<0));
angles(logical(angles>180)) = 360-angles(logical(angles>180));
ElevationAngles = angles;

end




