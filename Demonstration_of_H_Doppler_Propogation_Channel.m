Num_of_frame_each_Doppler_shift = 5000;
Parameter.parameters_doppler

MSE_LS_over_Doppler = zeros(length(DopplerShift), 1);
MSE_DDCE_over_Doppler = zeros(length(DopplerShift), 1);
MSE_MMSE_over_Doppler = zeros(length(DopplerShift), 1);
MSE_MME_over_Doppler = zeros(length(DopplerShift), 1);
MSE_DNN_over_Doppler = zeros(length(DopplerShift), 1);
MSE_ResNet_over_Doppler = zeros(length(DopplerShift), 1);
MSE_Transformer_over_Doppler = zeros(length(DopplerShift), 1);
MSE_HA02_over_Doppler = zeros(length(DopplerShift), 1);
MSE_Hybrid_over_Doppler = zeros(length(DopplerShift), 1);
MSE_Hybrid_frame_over_Doppler = zeros(length(DopplerShift), 1);
MSE_HA03_frame_over_Doppler = zeros(72, 14, length(DopplerShift));

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

for doppler = DopplerShift

Num_of_QPSK_symbols = Num_of_FFT * Num_of_symbols * Num_of_frame_each_Doppler_shift;
Num_of_bits = Num_of_QPSK_symbols * k;

LS_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
DDCE_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
MMSE_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
MME_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
DNN_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
ResNet_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
Transformer_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
HA02_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
Hybrid_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
Hybrid_frame_MSE_in_frame = zeros(Num_of_frame_each_Doppler_shift, 1);
MSE_HA03_frame = zeros(Num_of_FFT, Frame_size, Num_of_frame_each_Doppler_shift);

for Frame = 1 : Num_of_frame_each_Doppler_shift

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

SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
%[Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_user_for_channel] = Channel.Propagation_Channel_Model(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, doppler, DelayProfile);
[Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_for_channel, Multitap_Channel_Signal_user_for_channel] = Channel.Propagation_Channel_Model_test(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, doppler, DelayProfile);

% OFDM Receiver
[Unrecovered_signal, RS_User] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
[H_LS_Frame, RS] = OFDM.OFDM_Receiver(Multitap_Channel_Signal_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);

[Received_pilot, ~] = OFDM.Pilot_extract(RS_User, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_Ref = Received_pilot ./ Pilot_value_user;

% Channel estimation

% Perfect knowledge on Channel

% LS
[Received_pilot_LS, ~] = OFDM.Pilot_extract(Unrecovered_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

H_LS = CSI.LS(Received_pilot_LS, Pilot_value_user);

H_LS_frame = imresize(H_LS, [Num_of_FFT, max(Pilot_location_symbols)]);
H_LS_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_LS_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_LS_frame = mean(abs(H_LS_frame - RS).^2, 'all');

% DD-CE

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

MSE_DDCE_frame = mean(abs(H_LS_frame_DDCE - RS).^2, 'all');

% MMSE

% linear MMSE

H_MMSE = CSI.MMSE(H_Ref, RS, Pilot_location, Pilot_location_symbols, Num_of_FFT, SNR_OFDM, Num_of_pilot, H_LS);

H_MMSE_frame = imresize(H_MMSE, [Num_of_FFT, max(Pilot_location_symbols)]);
H_MMSE_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_MMSE_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

%H_MMSE_frame = CSI.MMSE_Interpolation(doppler, SampleRate, length_of_symbol, Frame_size, H_Ref, RS, Pilot_location_symbols, Num_of_FFT, SNR, Num_of_pilot, H_LS);

MSE_MMSE_frame = mean(abs(H_MMSE_frame - RS).^2, 'all');

% MMSE

H_MMSE_2D = CSI.MMSE_Interpolation_2D(RS, Pilot_location, Num_of_FFT, SNR, Frame_size, H_LS_Frame);

MSE_MMSE_2D_frame = mean(abs(H_MMSE_2D - RS).^2, 'all');

% Deep learning

Res_feature_signal(:, :, 1) = real(H_LS);
Res_feature_signal(:, :, 2) = imag(H_LS);

H_DNN_feature = predict(DNN_Trained, Res_feature_signal);

H_DNN = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);

H_DNN_frame = imresize(H_DNN, [Num_of_FFT, max(Pilot_location_symbols)]);
H_DNN_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_DNN_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_DNN_frame = mean(abs(H_DNN_frame - RS).^2, 'all');

% Deep learning

%tStart_ResNet = tic;

Res_feature_signal(:, :, 1) = real(H_LS);
Res_feature_signal(:, :, 2) = imag(H_LS);

H_ResNet_feature = predict(DNN_Trained_ResNetB, Res_feature_signal);

H_ResNet = H_ResNet_feature(:, :, 1) + 1j * H_ResNet_feature(:, :, 2);

H_ResNet_frame = imresize(H_ResNet, [Num_of_FFT, max(Pilot_location_symbols)]);
H_ResNet_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_ResNet_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

%t_ResNet(Frame, 1) = toc(tStart_ResNet);

MSE_ResNet_frame = mean(abs(H_ResNet_frame - RS).^2, 'all');

% Transformer

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Transformer_feature = transformer_HA02.model_transformer(Feature_signal, parameters_Transformer);

H_Transformer = reshape(extractdata(H_Transformer_feature(:, 1)), size(Pilot_location, 1), size(Pilot_location, 2)) + 1j * reshape(extractdata(H_Transformer_feature(:, 2)), size(Pilot_location, 1), size(Pilot_location, 2));

H_Transformer_frame = imresize(H_Transformer, [Num_of_FFT, max(Pilot_location_symbols)]);
H_Transformer_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_Transformer_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_Transformer_frame = mean(abs(H_Transformer_frame - RS).^2, 'all');

% HA02

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_HA02_feature = transformer_HA02.model(Feature_signal, parameters_ETU_HA02);

H_HA02_frame = reshape(extractdata(H_HA02_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_HA02_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_HA02_frame = mean(abs(H_HA02_frame - RS).^2, 'all');

% Hybrid structure

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Hybrid_feature = transformer.model(Feature_signal, parameters_frame);

H_Hybrid_frame_frame = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Frame_size) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Frame_size);

MSE_Hybrid_frame_frame = mean(abs(H_Hybrid_frame_frame - RS).^2, 'all');

MSE_HA03 = abs(H_Hybrid_frame_frame - RS).^2; 

% Hybrid structure

Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
Feature_signal = dlarray(Feature_signal);

H_Hybrid_feature = transformer.model(Feature_signal, parameters);

H_Hybrid_structure = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Num_of_pilot) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Num_of_pilot);

H_Hybrid_frame = imresize(H_Hybrid_structure, [Num_of_FFT, max(Pilot_location_symbols)]);
H_Hybrid_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_Hybrid_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

MSE_Hybrid_frame = mean(abs(H_Hybrid_frame - RS).^2, 'all');

% LS MSE calculation in each frame
LS_MSE_in_frame(Frame, 1) = MSE_LS_frame;

% DD-CE MSE calculation in each frame
DDCE_MSE_in_frame(Frame, 1) = MSE_DDCE_frame;

% MMSE MSE calculation in each frame
MMSE_MSE_in_frame(Frame, 1) = MSE_MMSE_frame;

% MMSE MSE calculation in each frame
MME_MSE_in_frame(Frame, 1) = MSE_MMSE_2D_frame;

% DNN MSE calculation in each frame
DNN_MSE_in_frame(Frame, 1) = MSE_DNN_frame;

% DNN MSE calculation in each frame
ResNet_MSE_in_frame(Frame, 1) = MSE_ResNet_frame;

% Transformer MSE calculation in each frame
Transformer_MSE_in_frame(Frame, 1) = MSE_Transformer_frame;

% Transformer MSE calculation in each frame
HA02_MSE_in_frame(Frame, 1) = MSE_HA02_frame;

% Hybrid MSE calculation in each frame
Hybrid_frame_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame_frame;

MSE_HA03_frame(:, :, Frame) = MSE_HA03;

% Hybrid MSE calculation in each frame
Hybrid_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame;

end

% MSE calculation

MSE_LS_over_Doppler(DopplerShift == doppler, 1) = sum(LS_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_DDCE_over_Doppler(DopplerShift == doppler, 1) = sum(DDCE_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_MMSE_over_Doppler(DopplerShift == doppler, 1) = sum(MMSE_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_MME_over_Doppler(DopplerShift == doppler, 1) = sum(MME_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_DNN_over_Doppler(DopplerShift == doppler, 1) = sum(DNN_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_ResNet_over_Doppler(DopplerShift == doppler, 1) = sum(ResNet_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_Transformer_over_Doppler(DopplerShift == doppler, 1) = sum(Transformer_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_HA02_over_Doppler(DopplerShift == doppler, 1) = sum(HA02_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_Hybrid_frame_over_Doppler(DopplerShift == doppler, 1) = sum(Hybrid_frame_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

MSE_HA03_frame_over_Doppler(:, :, DopplerShift == doppler) = sum(MSE_HA03_frame, 3) / Num_of_frame_each_Doppler_shift;

MSE_Hybrid_over_Doppler(DopplerShift == doppler, 1) = sum(Hybrid_MSE_in_frame, 1) / Num_of_frame_each_Doppler_shift;

end
