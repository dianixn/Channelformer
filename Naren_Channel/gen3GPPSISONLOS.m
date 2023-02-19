function h = gen3GPPSISONLOS(fc,numPath,velocityUE,t)

%  environment setup 
dis2D = 50; % BS-UE distance in horizontal plane, meter
waveLength = physconst('LightSpeed')/fc; 
BS_Location = [0;0;25];
UE_height = 1.5; 
numSector = 3; % BS sectorization
polarizationAngle = 0; % Vertically-polarized antennas at BS/UE

% geometry info
macroCellPar = getMacroCellPar(dis2D,BS_Location(3),UE_height);
UE_Location = macroCellPar.UE_Location;
dis3D = macroCellPar.dis3D;
ZOD = macroCellPar.ZOD;
AOD = macroCellPar.AOD;
ZOA = macroCellPar.ZOA;
AOA = macroCellPar.AOA;

% BS sector
sectorBS = getSectorBS(AOD);    

% antenna config - SISO
% BS
Nt = 1;
antPosBS = zeros(3,Nt); % 3 x Nt
antPosBS = getAntennaPosGCS(antPosBS,numSector,BS_Location); % 3 x [Nt Nt Nt]
orientationBS = [30 150 -90;0 0 0;0 0 0]; % 3 x NumSector, 3 x 3
posAntennaBS = antPosBS(:,Nt*(sectorBS-1)+1:Nt*sectorBS); % 3 x Nt

% UE
Nr = 1;
antPosUE = zeros(3,Nr); % 3 x Nr
orientationUE = zeros(3,1); 
posAntennaUE = antPosUE+repmat(UE_Location,1,Nr); % 3 x Nr

% generate time-domain channel
h = getChannelMatrixNLOS(fc,dis2D,velocityUE,t,waveLength,Nt,Nr,polarizationAngle,orientationBS,orientationUE,...
    posAntennaBS,posAntennaUE,sectorBS,UE_Location,numPath,dis3D,ZOD,AOD,ZOA,AOA);

