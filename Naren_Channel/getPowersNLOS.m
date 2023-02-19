function PowersNLOS = getPowersNLOS(DelaysNLOS,NumClusterNLOS,DS)
%%% This function is to calculate cluster delays of L NLOS links. 
% 
% Input 
%       DelaysNLOS       :  cluster delays, N x L
%       NumClusterNLOS   :  number of clusters N
%       DS               :  delay spread, 1 x L
%
% Output
%      PowersNLOS        :  cluster power, N x L
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r_tau = 2.3;
PerClusterSha = 3; % dB

DS = repmat(DS,NumClusterNLOS,1); % N x L

power_prime = exp(-((r_tau-1)/r_tau)*(DelaysNLOS./DS)).*(10.^(-normrnd(0,PerClusterSha,size(DS))./10)); 
PowersNLOS = power_prime./repmat(sum(power_prime,1),NumClusterNLOS,1); 
