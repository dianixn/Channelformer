function macroCellPar = getMacroCellPar(dis2D,BS_height,UE_height)

% 3D BS-UE distance
dis3D = sqrt((BS_height-UE_height)^2+dis2D^2);

% ZOA,ZOD -- [0 179] deg
ZOA = rand(size(dis2D))*90; % UE lower than BS
ZOD = 180-ZOA;

% AOA,AOD -- [-179 180] deg
AOD = (rand(size(dis2D))-0.5)*360;
AOA = zeros(size(dis2D));
AOA(logical(AOD>0)) = AOD(logical(AOD>0))-180;
AOA(logical(AOD<0)) = AOD(logical(AOD<0))+180;

% UE location
x_UE = dis2D.*cosd(AOD); 
y_UE = dis2D.*sind(AOD); 
z_UE = UE_height.*ones(size(dis2D)); 
UE_Location = [x_UE;y_UE;z_UE]; % 3 x 1

% LOS state
% LOS_probability = getProbabilityLOS_MacroCell(dis2D,UE_height); 
% LOS_State = binornd(1,LOS_probability); % 1 - LOS, 0 - NLOS
% LOS_State = zeros(size(LOS_probability));
% NumClusters(logical(LOS_State == 1)) = 12; % LOS
% NumClusters(logical(LOS_State == 0)) = 20; % NLOS


% Output
macroCellPar = struct('dis3D',dis3D,...
                      'UE_Location',UE_Location,...
                      'ZOD',ZOD,...
                      'AOD',AOD,...
                      'ZOA',ZOA,...
                      'AOA',AOA);
