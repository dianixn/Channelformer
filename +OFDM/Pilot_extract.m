function [Received_Pilot, Data_out] = Pilot_extract(Received_Data, Pilot_location, Num_of_pilot, Pilot_location_symbols, Data_location)

Data_out = Received_Data(:, Data_location);
Received_Pilot = zeros(size(Pilot_location, 1), Num_of_pilot);

for Pilot_location_symbol = Pilot_location_symbols
    Received_Pilot(:, Pilot_location_symbol == Pilot_location_symbols) = Received_Data(Pilot_location(:, Pilot_location_symbol == Pilot_location_symbols), Pilot_location_symbol);
end