function AntennaPosGCS = getAntennaPosGCS(AntPos,NumSector,BS_Location)

Nt = size(AntPos,2);
AntennaPosGCS = zeros(3,Nt,NumSector);

Bearing = [30 150 -90]; % the bearing angles of 3 sectors

for sec = 1:NumSector
    
    AntennaPosGCS(1,:,sec) = AntPos(2,:)*sind(-Bearing(sec));
    AntennaPosGCS(2,:,sec) = AntPos(2,:)*cosd(-Bearing(sec));
    AntennaPosGCS(3,:,sec) = AntPos(3,:);
    
end

AntennaPosGCS = reshape(AntennaPosGCS,3,Nt*NumSector); % 3 x [NtSec1 NtSec2 NtSec3]


% Include the BS location

AntennaPosGCS = AntennaPosGCS+repmat(BS_Location,1,Nt*NumSector);
