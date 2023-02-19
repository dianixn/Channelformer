function [ClusterZOA,RayZOA] = getAnglesZOA_NLOS(PowersNLOS,NumClusterNLOS,ZSA,ZOA,RayOffset,C_Ele_NLOS)
%%% This function is to calculate 
%                                 1. cluster zenith angle of arrival (ZOA) of L NLOS links; 
%                                 2. subpath zenith angle of arrival (ZOA) of L NLOS links. 
%
% Input 
%       PowersNLOS        :   cluster power, N x L
%       NumClusterNLOS    :   number of clusters, N
%       ZSA               :   azimuth spread of arrival, 1 x L
%       ZOA               :   LOS azimuth angle of arrival of direct paths, 1 x L
%       RayOffset         :   subpath angle offset, M x 1
%       C_Ele_NLOS        :   scaling factor, 1 x L
% Output
%       ClusterZOA        :   cluster ZOA, N x L
%       RayZOA            :   subpath ZOA, (M x N) x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(RayOffset); % M
NumNLOS = length(ZSA); % L

% Adjust vector dimensions, N x L.
ZSA = repmat(ZSA,NumClusterNLOS,1); 
ZOA = repmat(ZOA,NumClusterNLOS,1); 
C_Ele_NLOS = repmat(C_Ele_NLOS,NumClusterNLOS,1); 

% Obtain cluster ZOA, N x L.
ClusteZOA_temp = getClusterZOA_NLOS(ZSA,ZOA,PowersNLOS,C_Ele_NLOS,NumClusterNLOS); 

% Obtain subpath ZOA, M x (N x L).
RayZOA_temp = getRayZOA_NLOS(ClusteZOA_temp,RayOffset); 

% Obtain cluster ZOA within [0 179] degree.
ClusterZOA = wrapElevation(ClusteZOA_temp); 

% Randomly coupling subpaths within each cluster, M x (N x L)
RayZOA = Coupling(RayZOA_temp,M,NumClusterNLOS*NumNLOS);

% (M x N) x L
RayZOA = reshape(RayZOA,M*NumClusterNLOS,NumNLOS); 

end

function ClusterZOA = getClusterZOA_NLOS(ZSA,ZOA,PowersNLOS,C_Ele_NLOS,NumClusterNLOS)
%%% This function is to calculate cluster zenith angle of arrival (ZOA) of L NLOS links; 
%
% Input 
%       ZSA              :   azimuth spread of arrival, 1 x L
%       ZOA              :   LOS azimuth angle of arrival of direct paths, 1 x L
%       PowersNLOS       :   cluster power, N x L
%       C_Ele_NLOS       :   scaling factor, 1 x L
%       NumClusterNLOS   :   number of clusters, N
% Output
%       ClusterZOA       :   cluster ZOA, N x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% N x L
ZOA_prime = -ZSA.*log(PowersNLOS./repmat(max(PowersNLOS,[],1),NumClusterNLOS,1))./C_Ele_NLOS;
X = sign(rand(size(PowersNLOS))-0.5);
Y = normrnd(0,ZSA./7,size(ZSA));

ClusterZOA = X.*ZOA_prime+Y+ZOA;

end
function RayZOA = getRayZOA_NLOS(ClusteZOA_temp,RayOffset)
%%% This function is to calculate subpath zenith angle of arrival (ZOA) of L NLOS links. 
% 
% Input 
%       ClusterZOA_temp          :   cluster ZOA,  N x L
%       RayOffset                :   subpath angle offset, M x 1
% Output
%       RayZOA                   :   subpath ZOA, M x (N x L)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c_ZSA = 7; % deg

NumCluster = size(ClusteZOA_temp,1);
NumNLOS = size(ClusteZOA_temp,2);
M = length(RayOffset);

ClusteZOA_temp = reshape(ClusteZOA_temp,1,NumCluster*NumNLOS); % 1 x (N x L)
ClusteZOA_temp = repmat(ClusteZOA_temp,M,1); % M x (N x L)
RayZOA_temp = ClusteZOA_temp+c_ZSA*repmat(RayOffset,1,NumCluster*NumNLOS);  % M x (N x L)
RayZOA = wrapElevation(RayZOA_temp);

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








