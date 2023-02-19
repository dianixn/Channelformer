function H_MMSE = MMSE_Interpolation_2D(RS, Pilot_location, Num_of_FFT, SNR, Frame_size, H_LS)

H_MMSE = zeros(Num_of_FFT, Frame_size);

for i = 1 : Frame_size
    
    if i <= 7
        H_pilot = RS(Pilot_location(:, 1), i); 
        Rhh = H_pilot * H_pilot';
        H_MMSE(:, i) = (RS(:, i) * (H_pilot')) * pinv(Rhh + (1 / 10^(SNR / 10)) * eye(size(Rhh, 1))) * H_LS(Pilot_location(:, 1), i);      
    else
        H_pilot = RS(Pilot_location(:, 2), i); 
        Rhh = H_pilot * H_pilot';
        H_MMSE(:, i) = (RS(:, i) * (H_pilot')) * pinv((Rhh) + (1 / 10^(SNR / 10)) * eye(size(Rhh, 1))) * H_LS(Pilot_location(:, 2), i);
    end
    
end