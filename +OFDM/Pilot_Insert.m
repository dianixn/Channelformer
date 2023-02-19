function [Data_out, Data_location] = Pilot_Insert(Pilot_value, Pilot_location_symbols, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal)

% Pilot inserted
Data_out = zeros(Num_of_FFT, Frame_size);

for Pilot_location_symbol = Pilot_location_symbols
    Pilot_location_frequency = Pilot_location(:, Pilot_location_symbol == Pilot_location_symbols);
    Data_out(Pilot_location_frequency, Pilot_location_symbol) = Pilot_value;
end

Data_location = 1 : Frame_size;
Data_location(Pilot_location_symbols) = [];

Data_out(:, Data_location) = QPSK_signal;
