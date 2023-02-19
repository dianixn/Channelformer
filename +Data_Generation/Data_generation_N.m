% Channel Regression
% Data generation

function [Xtraining_Array_RSRP, Ytraining_regression_double_RSRP, Xvalidation_RSRP, Yvalidation_regression_double_RSRP] = Data_generation_N(Training_set_ratio, SNR_Range, Num_of_frame_each_SNR, type)

Parameter.parameters

Xtraining_Array_RSRP = zeros(size(Pilot_location, 1) * size(Pilot_location, 2), 2, 1, Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Ytraining_regression_double_RSRP = zeros(Num_of_FFT * Num_of_pilot, 2, 1, Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Xvalidation_RSRP = zeros(size(Pilot_location, 1) * size(Pilot_location, 2), 2, 1, Num_of_frame_each_SNR * size(SNR_Range, 2) - Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Yvalidation_regression_double_RSRP = zeros(Num_of_FFT * Num_of_pilot, 2, 1, Num_of_frame_each_SNR * size(SNR_Range, 2) - Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));

for SNR = SNR_Range
        
for Frame = 1 : Num_of_frame_each_SNR

%% Data generation

% Data generation
N = Num_of_FFT * Num_of_symbols;
data = randi([0 1], N, k);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = OFDM.QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_FFT, Num_of_symbols);
QPSK_signal_for_channel = ones(Num_of_FFT, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = OFDM.Pilot_Insert(Pilot_value_user, Pilot_location_symbols, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = OFDM.Pilot_Insert(1, Pilot_location_symbols, kron((1 : Num_of_FFT)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, QPSK_signal_for_channel);

% OFDM Transmitter
[Transmitted_signal, ~] = OFDM.OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM.OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

%% Channel

% AWGN Channel
        speedUE = randi([0, floor(50/3.6)]);
        directionUE = -90; % horizontal direction
        t = 0.01; % second, time interval
        numPath = 20;
        velocityUE = speedUE*[cosd(directionUE);sind(directionUE);0]; % 3 x 1

            h = gen3GPPSISONLOS(Carrier_Frequency,numPath,velocityUE,t); % Nr x Nt x numPath
            h = squeeze(h) / sqrt(numPath);
            
            Fading_signal_noise_free = conv(h,Transmitted_signal);
            Multitap_Channel_Signal_user = Fading_signal_noise_free(1:length(Transmitted_signal)); % noise-free receive data

            SignalPower = mean(abs(Multitap_Channel_Signal_user) .^ 2);
            Noise_Variance = SignalPower / (10 ^ (SNR / 10));

            Nvariance = sqrt(Noise_Variance / 2);
            n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

            Multitap_Channel_Signal = Multitap_Channel_Signal_user + n;

            Multitap_Channel = conv(h,Transmitted_signal_for_channel);
            Reference_Signal = Multitap_Channel(1:length(Transmitted_signal_for_channel)); % noise-free receive data

            SignalPower = mean(abs(Reference_Signal) .^ 2);
            Noise_Variance = SignalPower / (10 ^ ((SNR + 5 )/ 10));

            Nvariance = sqrt(Noise_Variance / 2);
            n = Nvariance * (randn(length(Transmitted_signal_for_channel), 1) + 1j * randn(length(Transmitted_signal_for_channel), 1)); % Noise generation
    

%% OFDM Receiver
[Received_signal, H_Ref] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Reference_Signal);

[Received_pilot, ~] = OFDM.Pilot_extract(Received_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_LS = Received_pilot / Pilot_value_user;

if Frame <= fix(Training_set_ratio * Num_of_frame_each_SNR)
    Training_index = Frame + fix(Training_set_ratio * Num_of_frame_each_SNR) * (find(SNR_Range == SNR) - 1);
    Xtraining_Array_RSRP(:, 1, 1, Training_index) = reshape(real(H_LS), [], 1);
    Xtraining_Array_RSRP(:, 2, 1, Training_index) = reshape(imag(H_LS), [], 1);
    Ytraining_regression_double_RSRP(:, 1, 1, Training_index) = reshape(real(H_Ref(:, Pilot_location_symbols)), [], 1);
    Ytraining_regression_double_RSRP(:, 2, 1, Training_index) = reshape(imag(H_Ref(:, Pilot_location_symbols)), [], 1);
else
    Validation_index = Frame - Training_set_ratio * Num_of_frame_each_SNR + (find(SNR_Range == SNR) - 1) * (Num_of_frame_each_SNR - Training_set_ratio * Num_of_frame_each_SNR);
    Xvalidation_RSRP(:, 1, 1, Validation_index) = reshape(real(H_LS), [], 1);
    Xvalidation_RSRP(:, 2, 1, Validation_index) = reshape(imag(H_LS), [], 1);
    Yvalidation_regression_double_RSRP(:, 1, 1, Validation_index) = reshape(real(H_Ref(:, Pilot_location_symbols)), [], 1);
    Yvalidation_regression_double_RSRP(:, 2, 1, Validation_index) = reshape(imag(H_Ref(:, Pilot_location_symbols)), [], 1);
end

end

end

end
