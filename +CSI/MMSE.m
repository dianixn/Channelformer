function H_MMSE = MMSE(H_Ref, RS, Pilot_location, Pilot_location_symbols, Num_of_FFT, SNR, Num_of_pilot, H_LS)

H_MMSE = zeros(Num_of_FFT, Num_of_pilot);

for i = 1 : size(Pilot_location, 2)
    H_pilot = H_Ref(:, i);
    Rhh = H_pilot * H_pilot';
    H_MMSE(:, i) = (RS(:, Pilot_location_symbols(i)) * (H_pilot')) * pinv(Rhh + (1 / 10^(SNR / 10)) * eye(size(Rhh, 1))) * H_LS(:, i);
end