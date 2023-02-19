function sectorBS = getSectorBS(AOD_LOS)

sectorBS = ones(size(AOD_LOS)).*2;

sectorBS(logical(AOD_LOS>=-30)&&logical(AOD_LOS<90)) = 1;
sectorBS(logical(AOD_LOS>=-150)&&logical(AOD_LOS<-30)) = 3;

end
