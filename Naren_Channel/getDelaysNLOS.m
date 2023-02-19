function DelaysNLOS = getDelaysNLOS(DS,NumClusterNLOS)
%%% This function is to calculate cluster delays of L NLOS links. 
% 
% Input 
%       DS               :  delay spread, 1 x L
%       NumClusterNLOS   :  number of clusters N
%
% Output
%      DelaysNLOS        :  cluster delays, N x L
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r_tau = 2.3;
DS = repmat(DS,NumClusterNLOS,1); % N x L

tau_prime = -r_tau*DS.*log(rand(size(DS))); % N x L
DelaysNLOS = sort(tau_prime-repmat(min(tau_prime,[],1),NumClusterNLOS,1),1); % N x L



