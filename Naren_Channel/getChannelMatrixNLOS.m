function ChannelMatrix = getChannelMatrixNLOS(fc,dis2D,velocityUE,t,WaveLength,Nt,Nr,polarizationAngle,orientationBS,orientationUE,PosAntennaBS,posAntennaUE,sectorBS,UE_Location,numPath,dis3D,ZOD,AOD,ZOA,AOA)

fc_GHz = fc*1e-9;

%%% 3GPP channel model

% Step 3: Path loss
[pathLoss,sha] = getPathLossNLOS(fc_GHz,dis3D); % NLOS path loss optional model
%pathLoss = pathLoss+normrnd(0,sha); % dB

% Step 4: Large-scale parameters
LSP = getLSP_NLOS(fc_GHz,dis2D,UE_Location(3),sha); % 7 x 1
SF = LSP(1);
DS = LSP(2);
ASD = LSP(3);
ASA = LSP(4);
ZSD = LSP(5);
ZSA = LSP(6);
mu_zsd = LSP(end);

% Step 5: Cluster delays
delays = getDelaysNLOS(DS,numPath); % N x 1

% Step 6: Cluster power
powers = getPowersNLOS(delays,numPath,DS); % N x 1

% Step 7 & 8: Cluster and subpath angles
% Subpath angle offset
rayOffset = [0.0447 0.1413 0.2492 0.3715 0.5129 0.6797 0.8844 1.1481 1.5195 2.1551];
rayOffset = [rayOffset;-rayOffset];
rayOffset = rayOffset(:); 

numSubPaths = length(rayOffset); % M

% Scaling factor
indexC_Azi = [4,5,8,10,11,12,14,15,16,19,20];
valueC_Azi = [0.779,0.860,1.018,1.090,1.123,1.146,1.190,1.211,1.226,1.273,1.289];
C_Azi_NLOS = valueC_Azi(logical(indexC_Azi==intersect(numPath,indexC_Azi)));

indexC_Ele = [8,10,11,12,15,19,20];
valueC_Ele = [0.889,0.957,1.031,1.104,1.1088,1.184,1.178];
C_Ele_NLOS = valueC_Ele(logical(indexC_Ele==intersect(numPath,indexC_Ele)));

% ClusterAOA/AOD/ZOA/ZOD: N x 1
% RayAOA/AOD/ZOA/ZOD: MN x 1
[clusterAOA,rayAOA] = getAnglesAOA_NLOS(powers,numPath,ASA,AOA,rayOffset,C_Azi_NLOS);
[clusterAOD,rayAOD] = getAnglesAOD_NLOS(powers,numPath,ASD,AOD,rayOffset,C_Azi_NLOS);
[clusterZOA,rayZOA] = getAnglesZOA_NLOS(powers,numPath,ZSA,ZOA,rayOffset,C_Ele_NLOS);
[clusterZOD,rayZOD] = getAnglesZOD_NLOS(powers,numPath,ZSD,ZOD,rayOffset,C_Ele_NLOS,mu_zsd,fc_GHz,dis2D,UE_Location(end));

angles = struct('ClusterAOA',clusterAOA,...
                'ClusterAOD',clusterAOD,...
                'ClusterZOA',clusterZOA,...
                'ClusterZOD',clusterZOD,...
                'RayAOA',rayAOA,...
                'RayAOD',rayAOD,...
                'RayZOA',rayZOA,...
                'RayZOD',rayZOD);
            
% Step 9: Cross polarization ratio            
XPR = getXPR_NLOS(numSubPaths,numPath,length(dis2D));   % 2 x 2 x MN
XPR = reshape(XPR,2,2*numSubPaths*numPath); % 2 x 2MN

% Step 10: Random phases
randomPhases = exp(1i*((rand(2,2*numSubPaths*numPath)-0.5)*2*pi)); % radian,2 x 2MN
middleMatrix = reshape(XPR.*randomPhases,2,2*numSubPaths,numPath); % 2 x 2M x N
middleMatrix = reshape(middleMatrix,2,2,numSubPaths,numPath); % 2 x 2 x M x N

% Step 10.5: Essential parameters for channel generations 

% Antenna pattern
AntennaPatternBS = getAntennaPatternBS(rayZOD,rayAOD,orientationBS,sectorBS,Nt,polarizationAngle,numSubPaths,numPath); % 2 x Nt x MN
AntennaPatternUE = getAntennaPatternUE(rayZOA,rayAOA,orientationUE,Nr,polarizationAngle); % 2 x Nr x MN

AntennaPatternBS = reshape(AntennaPatternBS,2,Nt,numSubPaths,numPath);
AntennaPatternUE = reshape(AntennaPatternUE,2,Nr,numSubPaths,numPath); 

% Spherical unit vectors, 3 x MN
sphericalUnitUE = SphericalUnitVector(rayZOA,rayAOA); % 3 x MN
sphericalUnitBS = SphericalUnitVector(rayZOD,rayAOD);

sphericalUnitUE = reshape(sphericalUnitUE,3,numSubPaths,numPath); % 3 x M x N
sphericalUnitBS = reshape(sphericalUnitBS,3,numSubPaths,numPath); % 3 x M x N

% Step 11:  Channel matrix
ChannelMatrix = zeros(Nr,Nt,numPath);

for n = 1:numPath
    
    FirstComponent = zeros(Nr,Nt,numSubPaths);
        
    for m = 1:numSubPaths % sum over M subpaths within per cluster
        
        % Antenna field pattern component, Nr x Nt
        % Weighted by the cluster power
        FirstComponent(:,:,m) = AntennaPatternUE(:,:,m,n).'*middleMatrix(:,:,m,n)*AntennaPatternBS(:,:,m,n);
    
    end
    FirstComponent = FirstComponent*sqrt(powers(n)/numSubPaths);
    FirstComponent = sum(FirstComponent,3);
    
    dopplerComponent = exp(1i*2*pi*sphericalUnitUE(:,:,n).'*velocityUE*t/WaveLength); % M x 1

    % Array response component, Nr x Nt
    SecondComponent = exp(1i*2*pi*sphericalUnitUE(:,:,n).'*posAntennaUE/WaveLength).'*diag(dopplerComponent)*...
        exp(1i*2*pi*sphericalUnitBS(:,:,n).'*PosAntennaBS/WaveLength);  
    
    % Channel matrix per cluster
    ChannelMatrix(:,:,n) = FirstComponent.*SecondComponent;

end


end
