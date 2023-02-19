function [ClusterAOD,RayAOD] = getAnglesAOD_NLOS(PowersNLOS,NumClusterNLOS,ASD,AOD,RayOffset,C_Azi_NLOS)
%%% This function is to calculate 
%                                 1. cluster azimuth angle of departure (AOD) of L NLOS links; 
%                                 2. subpath azimuth angle of departure (AOD) of L NLOS links. 
%
% Input 
%       PowersNLOS        :   cluster power, N x L
%       NumClusterNLOS    :   number of clusters, N
%       ASD               :   azimuth spread of departure, 1 x L
%       AOD               :   LOS azimuth angle of departure of direct paths, 1 x L
%       RayOffset         :   subpath angle offset, M x 1
%       C_Azi_NLOS        :   scaling factor, 1 x L
% Output
%       ClusterAOD       :   cluster AOD, N x L
%       RayAOD           :   subpath AOD, (M x N) x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(RayOffset); % M
NumNLOS = length(ASD); % L

% Adjust vector dimensions, N x L.
ASD = repmat(ASD,NumClusterNLOS,1); 
AOD = repmat(AOD,NumClusterNLOS,1); 
C_Azi_NLOS = repmat(C_Azi_NLOS,NumClusterNLOS,1); 

% Obtain cluster AOD, N x L.
ClusterAOD_temp = getClusterAOD_NLOS(ASD,AOD,PowersNLOS,C_Azi_NLOS,NumClusterNLOS); 

% Obtain subpath AOD, M x (N x L).
RayAOD_temp = getRayAOD_NLOS(ClusterAOD_temp,RayOffset); 

% Obtain cluster AOD within [-179 180] degree.
ClusterAOD = wrapAzimuth(ClusterAOD_temp); 

% Randomly coupling subpaths within each cluster, M x (N x L)
RayAOD = Coupling(RayAOD_temp,M,NumClusterNLOS*NumNLOS);

% (M x N) x L
RayAOD = reshape(RayAOD,M*NumClusterNLOS,NumNLOS);

end

function ClusterAOD = getClusterAOD_NLOS(ASD,AOD,PowersNLOS,C_Azi_NLOS,NumClusterNLOS)
%%% This function is to calculate cluster azimuth angle of departure (AOD) of L NLOS links.
%
% Input 
%       ASD              :   azimuth spread of departure, 1 x L
%       AOD              :   LOS azimuth angle of departure of direct paths, 1 x L
%       PowersNLOS       :   cluster power, N x L
%       C_Azi_NLOS       :   scaling factor, 1 x L
%       NumClusterNLOS   :   number of clusters, N
% Output
%       ClusterAOD       :   cluster AOD, N x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% N x L

ClusterAOD_prime = (2./C_Azi_NLOS).*(ASD./1.4).*sqrt(-log(PowersNLOS./repmat(max(PowersNLOS,[],1),NumClusterNLOS,1))); 
X = sign(rand(size(PowersNLOS))-0.5);
Y = normrnd(0,ASD./7,size(ASD));

ClusterAOD = X.*ClusterAOD_prime+Y+AOD; 

end
function RayAOD = getRayAOD_NLOS(ClusterAOD_temp,RayOffset)
%%% This function is to calculate subpath azimuth angle of departure (AOD) of L NLOS links. 
% 
% Input 
%       ClusterAOD_temp          :   cluster AOD,  N x L
%       RayOffset                :   subpath angle offset, M x 1
% Output
%       RayAOD                   :   subpath AOD, M x (N x L)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c_ASD = 2; % deg
NumCluster = size(ClusterAOD_temp,1);
NumNLOS = size(ClusterAOD_temp,2);
M = length(RayOffset);

ClusterAOD_temp = reshape(ClusterAOD_temp,1,NumCluster*NumNLOS); % 1 x (N x L)
ClusterAOD_temp = repmat(ClusterAOD_temp,M,1); % M x (N x L)
RayAOD_temp = ClusterAOD_temp+c_ASD*repmat(RayOffset,1,NumCluster*NumNLOS);  % M x (N x L)

RayAOD = wrapAzimuth(RayAOD_temp);

end

function AzimuthAngles = wrapAzimuth(angles)
%%% [-179,180]

angles(logical(angles>180)) = angles(logical(angles>180))-360;
angles(logical(angles<-180)) = angles(logical(angles<-180))+360;
AzimuthAngles = angles;

end


function Angles_ray = Coupling(Angles_ray_temp,M,N)
%%% Coupling subpaths within one cluster

[~,order] = sort(rand(M,N),1);
index = order+repmat([1:M:M*N],M,1)-1;
Angles_ray_temp = Angles_ray_temp(index);
Angles_ray = reshape(Angles_ray_temp,M,N);

end






