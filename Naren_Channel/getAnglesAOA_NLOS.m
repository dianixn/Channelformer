function [ClusterAOA,RayAOA] = getAnglesAOA_NLOS(PowersNLOS,NumClusterNLOS,ASA,AOA,RayOffset,C_Azi_NLOS)
%%% This function is to calculate 
%                                 1. cluster azimuth angle of arrival (AOA) of L NLOS links; 
%                                 2. subpath azimuth angle of arrival (AOA) of L NLOS links. 
%
% Input 
%       PowersNLOS        :   cluster power, N x L
%       NumClusterNLOS    :   number of clusters, N
%       ASA               :   azimuth spread of arrival, 1 x L
%       AOA               :   LOS azimuth angle of arrival of direct paths, 1 x L
%       RayOffset         :   subpath angle offset, M x 1
%       C_Azi_NLOS         :   scaling factor, 1 x L
% Output
%       ClusterAOA        :   cluster AOA, N x L
%       RayAOA            :   subpath AOA, (M x N) x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(RayOffset); % M
NumNLOS = length(ASA); % L

% Adjust variable dimensions, N x L.
ASA = repmat(ASA,NumClusterNLOS,1); 
AOA = repmat(AOA,NumClusterNLOS,1); 
C_Azi_NLOS = repmat(C_Azi_NLOS,NumClusterNLOS,1);

% Obtain cluster AOA, N x L.
ClusterAOA_temp = getClusterAOA_NLOS(ASA,AOA,PowersNLOS,C_Azi_NLOS,NumClusterNLOS); 

% Obtain subpath AOA, M x (N x L).
RayAOA_temp = getRayAOA_NLOS(ClusterAOA_temp,RayOffset); 

% Obtain cluster AOA within [-179 180] degree.
ClusterAOA = wrapAzimuth(ClusterAOA_temp); 

% Randomly coupling subpaths within each cluster, M x (N x L)
RayAOA = Coupling(RayAOA_temp,M,NumClusterNLOS*NumNLOS);

% (M x N) x L
RayAOA = reshape(RayAOA,M*NumClusterNLOS,NumNLOS); 

end

function ClusterAOA = getClusterAOA_NLOS(ASA,AOA,PowersNLOS,C_Azi_NLOS,NumClusterNLOS)
%%% This function is to calculate cluster azimuth angle of arrival (AOA) of L NLOS links; 
%
% Input 
%       ASA              :   azimuth spread of arrival, 1 x L
%       AOA .            :   LOS azimuth angle of arrival of direct paths, 1 x L
%       PowersNLOS       :   cluster power, N x L
%       C_Azi_NLOS       :   scaling factor, 1 x L
%       NumClusterNLOS   :   number of clusters, N
% Output
%       ClusterAOA       :   cluster AOA, N x L
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% N x L
ClusterAOA_prime = (2./C_Azi_NLOS).*(ASA./1.4).*sqrt(-log(PowersNLOS./repmat(max(PowersNLOS,[],1),NumClusterNLOS,1))); 
X = sign(rand(size(PowersNLOS))-0.5);
Y = normrnd(0,ASA./7,size(ASA));

ClusterAOA = X.*ClusterAOA_prime+Y+AOA; 

end

function RayAOA = getRayAOA_NLOS(ClusterAOA_temp,RayOffset)
%%% This function is to calculate subpath azimuth angle of arrival (AOA) of L NLOS links. 
% 
% Input 
%       ClusterAOA_temp          :   cluster AOA,  N x L
%       RayOffset                :   subpath angle offset, M x 1
% Output
%       RayAOA                   :   subpath AOA, M x (N x L)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c_ASA = 15; % deg

NumCluster = size(ClusterAOA_temp,1); % N
NumNLOS = size(ClusterAOA_temp,2); % L
M = length(RayOffset); % M

ClusterAOA_temp = reshape(ClusterAOA_temp,1,NumCluster*NumNLOS); % 1 x (N x L)
ClusterAOA_temp = repmat(ClusterAOA_temp,M,1); % M x (N x L)
RayAOA_temp = ClusterAOA_temp+c_ASA*repmat(RayOffset,1,NumCluster*NumNLOS);  % M x (N x L)

RayAOA = wrapAzimuth(RayAOA_temp);
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






