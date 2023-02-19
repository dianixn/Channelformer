SNR_Range = -10:5:30;
Num_of_frame_each_SNR = 5000;

MSE_LS_over_SNR = zeros(length(SNR_Range), 1);
MSE_DDCE_over_SNR = zeros(length(SNR_Range), 1);
MSE_MMSE_over_SNR = zeros(length(SNR_Range), 1);
MSE_MME_over_SNR = zeros(length(SNR_Range), 1);
MSE_DNN_over_SNR = zeros(length(SNR_Range), 1);
MSE_Transformer_over_SNR = zeros(length(SNR_Range), 1);
MSE_ResNet_over_SNR = zeros(length(SNR_Range), 1);
MSE_HA02_over_SNR = zeros(length(SNR_Range), 1);
MSE_Hybrid_frame_over_SNR = zeros(length(SNR_Range), 1);
MSE_HA03_frame_over_SNR = zeros(72, 14, length(SNR_Range));
MSE_Interpolation_frame_over_SNR = zeros(72, 14, length(SNR_Range));
MSE_Hybrid_over_SNR = zeros(length(SNR_Range), 1);

BER_LS_over_SNR = zeros(length(SNR_Range), 1);
BER_DDCE_over_SNR = zeros(length(SNR_Range), 1);
BER_MMSE_over_SNR = zeros(length(SNR_Range), 1);
BER_MME_over_SNR = zeros(length(SNR_Range), 1);
BER_DNN_over_SNR = zeros(length(SNR_Range), 1);
BER_ResNet_over_SNR = zeros(length(SNR_Range), 1);
BER_Transformer_over_SNR = zeros(length(SNR_Range), 1);
BER_HA02_over_SNR = zeros(length(SNR_Range), 1);
BER_Hybrid_frame_over_SNR = zeros(length(SNR_Range), 1);
BER_Hybrid_over_SNR = zeros(length(SNR_Range), 1);

T_LS = zeros(length(SNR_Range), 1);
T_DDCE = zeros(length(SNR_Range), 1);
T_MMSE = zeros(length(SNR_Range), 1);
T_MME = zeros(length(SNR_Range), 1);
T_DNN = zeros(length(SNR_Range), 1);
T_ResNet = zeros(length(SNR_Range), 1);
T_Transformer = zeros(length(SNR_Range), 1);
T_HA02 = zeros(length(SNR_Range), 1);
T_Hybrid = zeros(length(SNR_Range), 1);
T_Hybrid_frame = zeros(length(SNR_Range), 1);

% Import Deep Neuron Network
load('Interpolation_ResNet.mat');

% Import Deep Neuron Network
load('ReEsNet.mat');

% Import Transformer Network
load('parameters_TR.mat');

% Import HA02 Network
load('parameters_ETU_HA02.mat');

% Import Hybrid Network
load('parameters_HA03.mat');

% Import Offline Hybrid Network
load('parameters_frame_3_decode.mat');

Parameter.parameters

%[RHh, Rhh] = CSI.Rhh_estimation();

for SNR = SNR_Range
    
M = 4; % QPSK
k = log2(M);

Num_of_QPSK_symbols = Num_of_FFT * Num_of_symbols * Num_of_frame_each_SNR;
Num_of_bits = Num_of_QPSK_symbols * k;

LS_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
DDCE_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MMSE_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MMSE_2D_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
DNN_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
ResNet_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Transformer_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
HA02_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Hybrid_frame_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MSE_Interpolation_frame = zeros(Num_of_FFT, Frame_size, Num_of_frame_each_SNR);
MSE_HA03_frame = zeros(Num_of_FFT, Frame_size, Num_of_frame_each_SNR);
Hybrid_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);

t = zeros(Num_of_frame_each_SNR, 1);
t_LS = zeros(Num_of_frame_each_SNR, 1);
t_DDCE = zeros(Num_of_frame_each_SNR, 1);
t_MMSE = zeros(Num_of_frame_each_SNR, 1);
t_MME = zeros(Num_of_frame_each_SNR, 1);
t_DNN = zeros(Num_of_frame_each_SNR, 1);
t_ResNet = zeros(Num_of_frame_each_SNR, 1);
t_Transformer = zeros(Num_of_frame_each_SNR, 1);
t_HA02 = zeros(Num_of_frame_each_SNR, 1);
t_Hybrid_frame = zeros(Num_of_frame_each_SNR, 1);
t_Hybrid = zeros(Num_of_frame_each_SNR, 1);

Frame_error_LS = 0;
Frame_error_DDCE = 0;
Frame_error_MMSE = 0;
Frame_error_MME = 0;
Frame_error_DNN = 0;
Frame_error_ResNet = 0;
Frame_error_Transformer = 0;
Frame_error_HA02 = 0;
Frame_error_Hybrid_frame = 0;
Frame_error_Hybrid = 0;

for Frame = 1 : Num_of_frame_each_SNR

% Data generation
N = Num_of_FFT * Num_of_symbols;
data = randi([0 1], N, k);
Data = reshape(data, [], 1);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = OFDM.QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_FFT, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = OFDM.Pilot_Insert(Pilot_value_user, Pilot_location_symbols, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = OFDM.Pilot_Insert(1, Pilot_location_symbols, kron((1 : Num_of_FFT)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, (ones(Num_of_FFT, Num_of_symbols)));

% OFDM Transmitter
[Transmitted_signal, ~] = OFDM.OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM.OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

% Channel

SNR_OFDM = SNR;
Doppler_shift = randi([0, MaxDopplerShift]);
%[Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_user_for_channel] = Channel.Propagation_Channel_Model(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, Doppler_shift, DelayProfile);
[Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_for_channel, Multitap_Channel_Signal_user_for_channel] = Channel.Propagation_Channel_Model_test(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, Doppler_shift, DelayProfile);

% OFDM Receiver
[Unrecovered_signal, RS_User] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
[H_LS_Frame, RS] = OFDM.OFDM_Receiver(Multitap_Channel_Signal_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);

[Received_pilot, ~] = OFDM.Pilot_extract(RS_User, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_Ref = Received_pilot ./ Pilot_value_user;

% Channel estimation

% Perfect knowledge on Channel

% LS
[Received_pilot_LS, ~] = OFDM.Pilot_extract(Unrecovered_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

tStart_LS = tic;

H_LS = CSI.LS(Received_pilot_LS, Pilot_value_user);

H_LS_frame = imresize(H_LS, [Num_of_FFT, max(Pilot_location_symbols)]);
H_LS_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_LS_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

t_LS(Frame, 1) = toc(tStart_LS);

MSE_LS_frame = mean(abs(H_LS_frame - RS).^2, 'all');

% DD-CE

tStart_DDCE = tic;

H_LS_frame_DDCE = H_LS_frame;

for i = 1:100

X = Unrecovered_signal ./ H_LS_frame_DDCE;
[row, col] = find((angle(X) > 0) .* (angle(X) < (pi/2)));
for j = 1:size(row, 1)
    X(row(j), col(j)) = 1+1j;
end
[row, col] = find((angle(X) > (pi/2)) .* (angle(X) < pi));
for j = 1:size(row, 1)
    X(row(j), col(j)) = -1+1j;
end
[row, col] = find((angle(X) > (- pi)) .* (angle(X) < (- pi/2)));
for j = 1:size(row, 1)
    X(row(j), col(j)) = -1-1j;
end
[row, col] = find((angle(X) > (- pi/2)) .* (angle(X) < 0));
for j = 1:size(row, 1)
    X(row(j), col(j)) = 1-1j;
end
H_LS_frame_DDCE = Unrecovered_signal ./ X; 

end

H_LS_frame_DDCE = CSI.MMSE_Uniform_PDP(length_of_CP, Num_of_FFT, 24, 10^(10/10), true, H_LS_frame_DDCE);

t_DDCE(Frame, 1) = toc(tStart_DDCE);

MSE_DDCE_frame = mean(abs(H_LS_frame_DDCE - RS).^2, 'all');

% MMSE

% linear MMSE

tStart_MMSE = tic;

H_MMSE = CSI.MMSE(H_Ref, RS, Pilot_location, Pilot_location_symbols, Num_of_FFT, SNR, Num_of_pilot, H_LS);

%H_MMSE = CSI.MMSE_Wiener(Num_of_FFT, 2, T_rms, SNR, H_LS);

H_MMSE_frame = imresize(H_MMSE, [Num_of_FFT, max(Pilot_location_symbols)]);
H_MMSE_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_MMSE_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

t_MMSE(Frame, 1) = toc(tStart_MMSE);

MSE_MMSE_frame = mean(abs(H_MMSE_frame - RS).^2, 'all');

% 2D MMSE

tStart_MMSE_2D = tic;

H_MMSE_2D = CSI.MMSE_Interpolation_2D(RS, Pilot_location, Num_of_FFT, SNR, Frame_size, H_LS_Frame);

t_MME(Frame, 1) = toc(tStart_MMSE_2D);

MSE_MMSE_2D_frame = mean(abs(H_MMSE_2D - RS).^2, 'all');

% Deep learning

tStart_DNN = tic;

Res_feature_signal(:, :, 1) = real(H_LS);
Res_feature_signal(:, :, 2) = imag(H_LS);

H_DNN_feature = predict(DNN_Trained, Res_feature_signal);

H_DNN_frame = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);

t_DNN(Frame, 1) = toc(tStart_DNN);

MSE_DNN_frame = mean(abs(H_DNN_frame - RS).^2, 'all');

% Deep learning

tStart_ResNet = tic;

Res_feature_signal(:, :, 1) = real(H_LS);
Res_feature_signal(:, :, 2) = imag(H_LS);

H_ResNet_feature = predict(DNN_Trained_ResNetB, Res_feature_signal);

H_ResNet_frame = H_ResNet_feature(:, :, 1) + 1j * H_ResNet_feature(:, :, 2);

t_ResNet(Frame, 1) = toc(tStart_ResNet);

MSE_ResNet_frame = mean(abs(H_ResNet_frame - RS).^2, 'all');

MSE_Interpolation = abs(H_ResNet_frame - RS).^2; 

% Transformer

tStart_Transformer = tic;

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Transformer_feature = transformer_HA02.model_transformer(Feature_signal, parameters_Transformer);

H_Transformer = reshape(extractdata(H_Transformer_feature(:, 1)), size(Pilot_location, 1), size(Pilot_location, 2)) + 1j * reshape(extractdata(H_Transformer_feature(:, 2)), size(Pilot_location, 1), size(Pilot_location, 2));

t_Transformer(Frame, 1) = toc(tStart_Transformer);

H_Transformer_frame = imresize(H_Transformer, [Num_of_FFT, max(Pilot_location_symbols)]);
H_Transformer_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_Transformer_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_Transformer_frame = mean(abs(H_Transformer_frame - RS).^2, 'all');

% HA02

tStart_HA02 = tic;

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_HA02_feature = transformer_HA02.model(Feature_signal, parameters_ETU_HA02);

H_HA02_frame = reshape(extractdata(H_HA02_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_HA02_feature(:, 2)), Num_of_FFT, Frame_size);

t_HA02(Frame, 1) = toc(tStart_HA02);

MSE_HA02_frame = mean(abs(H_HA02_frame - RS).^2, 'all');

% Hybrid structure

tStart_Hybrid_frame = tic;

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Hybrid_feature = transformer.model(Feature_signal, parameters_frame);

H_Hybrid_frame_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

t_Hybrid_frame(Frame, 1) = toc(tStart_Hybrid_frame);

MSE_Hybrid_frame_frame = mean(abs(H_Hybrid_frame_frame - RS).^2, 'all');

MSE_HA03 = abs(H_Hybrid_frame_frame - RS).^2; 

% Hybrid structure

tStart_Hybrid = tic;

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Hybrid_feature = transformer.model(Feature_signal, parameters);

H_Hybrid_structure = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Num_of_pilot) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Num_of_pilot);

t_Hybrid(Frame, 1) = toc(tStart_Hybrid);

H_Hybrid_frame = imresize(H_Hybrid_structure, [Num_of_FFT, max(Pilot_location_symbols)]);
H_Hybrid_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_Hybrid_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_Hybrid_frame = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% QPSK demodulation

LS_recover_signal = Unrecovered_signal .* conj(H_LS_frame) ./ (abs(H_LS_frame) .^ 2);
DDCE_recover_signal = Unrecovered_signal .* conj(H_LS_frame_DDCE) ./ (abs(H_LS_frame_DDCE) .^ 2);
MMSE_recover_signal = Unrecovered_signal .* conj(H_MMSE_frame) ./ (abs(H_MMSE_frame) .^ 2);
MME_recover_signal = Unrecovered_signal .* conj(H_MMSE_2D) ./ (abs(H_MMSE_2D) .^ 2);
DNN_recover_signal = Unrecovered_signal .* conj(H_DNN_frame) ./ (abs(H_DNN_frame) .^ 2);
ResNet_recover_signal = Unrecovered_signal .* conj(H_ResNet_frame) ./ (abs(H_ResNet_frame) .^ 2);
Transformer_recover_signal = Unrecovered_signal .* conj(H_Transformer_frame) ./ (abs(H_Transformer_frame) .^ 2);
HA02_recover_signal = Unrecovered_signal .* conj(H_HA02_frame) ./ (abs(H_HA02_frame) .^ 2);
Hybrid_frame_recover_signal = Unrecovered_signal .* conj(H_Hybrid_frame_frame) ./ (abs(H_Hybrid_frame_frame) .^ 2);
Hybrid_recover_signal = Unrecovered_signal .* conj(H_Hybrid_frame) ./ (abs(H_Hybrid_frame) .^ 2);

[~, LS_recover_data] = OFDM.Pilot_extract(LS_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, DDCE_recover_data] = OFDM.Pilot_extract(DDCE_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, MMSE_recover_data] = OFDM.Pilot_extract(MMSE_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, MME_recover_data] = OFDM.Pilot_extract(MME_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, DNN_recover_data] = OFDM.Pilot_extract(DNN_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, ResNet_recover_data] = OFDM.Pilot_extract(ResNet_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, Transformer_recover_data] = OFDM.Pilot_extract(Transformer_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, HA02_recover_data] = OFDM.Pilot_extract(HA02_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
[~, Hybrid_frame_recover_data] = OFDM.Pilot_extract(Hybrid_frame_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

%Hybrid_frame_recover_data(end - 2 : end, 12) = DNN_recover_data(end - 2 : end, 12);

[~, Hybrid_recover_data] = OFDM.Pilot_extract(Hybrid_recover_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

LS_dataSym = OFDM.QPSK_Demodulator(reshape(LS_recover_data, [], 1));
DDCE_dataSym = OFDM.QPSK_Demodulator(reshape(DDCE_recover_data, [], 1));
MMSE_dataSym = OFDM.QPSK_Demodulator(reshape(MMSE_recover_data, [], 1));
MME_dataSym = OFDM.QPSK_Demodulator(reshape(MME_recover_data, [], 1));
DNN_dataSym = OFDM.QPSK_Demodulator(reshape(DNN_recover_data, [], 1));
ResNet_dataSym = OFDM.QPSK_Demodulator(reshape(ResNet_recover_data, [], 1));
Transformer_dataSym = OFDM.QPSK_Demodulator(reshape(Transformer_recover_data, [], 1));
HA02_dataSym = OFDM.QPSK_Demodulator(reshape(HA02_recover_data, [], 1));
Hybrid_frame_dataSym = OFDM.QPSK_Demodulator(reshape(Hybrid_frame_recover_data, [], 1));
Hybrid_dataSym = OFDM.QPSK_Demodulator(reshape(Hybrid_recover_data, [], 1));

Data_LS = reshape(de2bi(LS_dataSym), [], 1);
Data_DDCE = reshape(de2bi(DDCE_dataSym), [], 1);
Data_MMSE = reshape(de2bi(MMSE_dataSym), [], 1);
Data_MME = reshape(de2bi(MME_dataSym), [], 1);
Data_DNN = reshape(de2bi(DNN_dataSym), [], 1);
Data_ResNet = reshape(de2bi(ResNet_dataSym), [], 1);
Data_Transformer = reshape(de2bi(Transformer_dataSym), [], 1);
Data_HA02 = reshape(de2bi(HA02_dataSym), [], 1);
Data_Hybrid_frame = reshape(de2bi(Hybrid_frame_dataSym), [], 1);
Data_Hybrid = reshape(de2bi(Hybrid_dataSym), [], 1);

Error_LS = sum(round(Data_LS) ~= round(Data));
Error_DDCE = sum(round(Data_DDCE) ~= round(Data));
Error_MMSE = sum(round(Data_MMSE) ~= round(Data));
Error_MME = sum(round(Data_MME) ~= round(Data));
Error_DNN = sum(round(Data_DNN) ~= round(Data));
Error_ResNet = sum(round(Data_ResNet) ~= round(Data));
Error_Transformer = sum(round(Data_Transformer) ~= round(Data));
Error_HA02 = sum(round(Data_HA02) ~= round(Data));
Error_Hybrid_frame = sum(round(Data_Hybrid_frame) ~= round(Data));
Error_Hybrid = sum(round(Data_Hybrid) ~= round(Data));

Frame_error_LS = Frame_error_LS + Error_LS;
Frame_error_DDCE = Frame_error_DDCE + Error_DDCE;
Frame_error_MMSE = Frame_error_MMSE + Error_MMSE;
Frame_error_MME = Frame_error_MME + Error_MME;
Frame_error_DNN = Frame_error_DNN + Error_DNN;
Frame_error_ResNet = Frame_error_ResNet + Error_ResNet;
Frame_error_Transformer = Frame_error_Transformer + Error_Transformer;
Frame_error_HA02 = Frame_error_HA02 + Error_HA02;
Frame_error_Hybrid_frame = Frame_error_Hybrid_frame + Error_Hybrid_frame;
Frame_error_Hybrid = Frame_error_Hybrid + Error_Hybrid;

% LS MSE calculation in each frame
LS_MSE_in_frame(Frame, 1) = MSE_LS_frame;

% DD-CE MSE calculation in each frame
DDCE_MSE_in_frame(Frame, 1) = MSE_DDCE_frame;

% MMSE MSE calculation in each frame
MMSE_MSE_in_frame(Frame, 1) = MSE_MMSE_frame;

% MMSE MSE calculation in each frame
MMSE_2D_MSE_in_frame(Frame, 1) = MSE_MMSE_2D_frame;

% DNN MSE calculation in each frame
DNN_MSE_in_frame(Frame, 1) = MSE_DNN_frame;

MSE_Interpolation_frame(:, :, Frame) = MSE_Interpolation;

% DNN MSE calculation in each frame
ResNet_MSE_in_frame(Frame, 1) = MSE_ResNet_frame;

% Transformer MSE calculation in each frame
Transformer_MSE_in_frame(Frame, 1) = MSE_Transformer_frame;

% HA02 MSE calculation in each frame
HA02_MSE_in_frame(Frame, 1) = MSE_HA02_frame;

% Hybrid MSE calculation in each frame
Hybrid_frame_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame_frame;

MSE_HA03_frame(:, :, Frame) = MSE_HA03;

% Hybrid MSE calculation in each frame
Hybrid_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame;

end

% MSE calculation
MSE_LS_over_SNR(SNR_Range == SNR, 1) = sum(LS_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_DDCE_over_SNR(SNR_Range == SNR, 1) = sum(DDCE_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_MMSE_over_SNR(SNR_Range == SNR, 1) = sum(MMSE_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_MME_over_SNR(SNR_Range == SNR, 1) = sum(MMSE_2D_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_DNN_over_SNR(SNR_Range == SNR, 1) = sum(DNN_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Interpolation_frame_over_SNR(:, :, SNR_Range == SNR) = sum(MSE_Interpolation_frame, 3) / Num_of_frame_each_SNR;

MSE_ResNet_over_SNR(SNR_Range == SNR, 1) = sum(ResNet_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Transformer_over_SNR(SNR_Range == SNR, 1) = sum(Transformer_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_HA02_over_SNR(SNR_Range == SNR, 1) = sum(HA02_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Hybrid_frame_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_frame_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_HA03_frame_over_SNR(:, :, SNR_Range == SNR) = sum(MSE_HA03_frame, 3) / Num_of_frame_each_SNR;

MSE_Hybrid_over_SNR(SNR_Range == SNR, 1) = sum(Hybrid_MSE_in_frame, 1) / Num_of_frame_each_SNR;

% BER calculation
BER_LS_over_SNR(SNR_Range == SNR, 1) = Frame_error_LS / (Num_of_frame_each_SNR * N * k);

BER_DDCE_over_SNR(SNR_Range == SNR, 1) = Frame_error_DDCE / (Num_of_frame_each_SNR * N * k);

BER_MMSE_over_SNR(SNR_Range == SNR, 1) = Frame_error_MMSE / (Num_of_frame_each_SNR * N * k);

BER_MME_over_SNR(SNR_Range == SNR, 1) = Frame_error_MME / (Num_of_frame_each_SNR * N * k);

BER_DNN_over_SNR(SNR_Range == SNR, 1) = Frame_error_DNN / (Num_of_frame_each_SNR * N * k);

BER_ResNet_over_SNR(SNR_Range == SNR, 1) = Frame_error_ResNet / (Num_of_frame_each_SNR * N * k);

BER_Transformer_over_SNR(SNR_Range == SNR, 1) = Frame_error_Transformer / (Num_of_frame_each_SNR * N * k);

BER_HA02_over_SNR(SNR_Range == SNR, 1) = Frame_error_HA02 / (Num_of_frame_each_SNR * N * k);

BER_Hybrid_frame_over_SNR(SNR_Range == SNR, 1) = Frame_error_Hybrid_frame / (Num_of_frame_each_SNR * N * k);

BER_Hybrid_over_SNR(SNR_Range == SNR, 1) = Frame_error_Hybrid / (Num_of_frame_each_SNR * N * k);

% Excution time

T_LS(SNR_Range == SNR, 1) = sum(t_LS, 1) / Num_of_frame_each_SNR;

T_DDCE(SNR_Range == SNR, 1) = sum(t_DDCE, 1) / Num_of_frame_each_SNR;

T_MMSE(SNR_Range == SNR, 1) = sum(t_MMSE, 1) / Num_of_frame_each_SNR;

T_MME(SNR_Range == SNR, 1) = sum(t_MME, 1) / Num_of_frame_each_SNR;

T_DNN(SNR_Range == SNR, 1) = sum(t_DNN, 1) / Num_of_frame_each_SNR;

T_ResNet(SNR_Range == SNR, 1) = sum(t_ResNet, 1) / Num_of_frame_each_SNR;

T_Transformer(SNR_Range == SNR, 1) = sum(t_Transformer, 1) / Num_of_frame_each_SNR;

T_HA02(SNR_Range == SNR, 1) = sum(t_HA02, 1) / Num_of_frame_each_SNR;

T_Hybrid(SNR_Range == SNR, 1) = sum(t_Hybrid, 1) / Num_of_frame_each_SNR;

T_Hybrid_frame(SNR_Range == SNR, 1) = sum(t_Hybrid_frame, 1) / Num_of_frame_each_SNR;

end
