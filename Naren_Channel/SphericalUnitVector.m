function UnitVector = SphericalUnitVector(theta,phi)

% theta:  X x 1
% phi : X x 1

UnitVector = [sind(theta.').*cosd(phi.');...
                sind(theta.').*sind(phi.');...
                cosd(theta.')]; % 3 x X

end

