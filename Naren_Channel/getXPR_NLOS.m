function XPR_Matrix = getXPR_NLOS(NumSubPaths,NumCluster,NumNLOS) 
%%% This function is to calculate the cross polarization power ratios (XPR) for each subpath in each cluster.
%
% Input
%       NumSubPaths    :    M
%       NumCluster     :    N
%       NumNLOS        :    L
% Output
%       XPR_Matrix     :    2 x 2 x (M x N) x L

mu_XPR = 7;
sigma_XPR = 3;

X = normrnd(mu_XPR,sigma_XPR,[1,NumSubPaths*NumCluster*NumNLOS]); % 1 x MN*NumNLOS
XPR = 10.^(0.1*X);

% Compute the matrix of square root reciprocal of XPR used in calculating polarized antenna pattern.
% 2 x (2 x M x N x L)
XPR_temp = repelem(sqrt(XPR.^(-1)),2,2); % % 2 x (2 x M x N x L)
XPR_temp(1:4:end) = 1;
XPR_temp(4:4:end) = 1;
XPR_Matrix = reshape(XPR_temp,2,2,NumSubPaths*NumCluster,NumNLOS); % 2 x 2 x MN x L

end





