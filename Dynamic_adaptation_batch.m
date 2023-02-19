SNR_Range = [15, 25];
Num_of_frame_each_SNR = 10000; 

Frame_size = 10; 

batch_size = 5;

Parameter.parameters

load('parameters_HA03.mat');

Threshold_Encoder_Ratio = 0.7;
Threshold_Decoder_Ratio = 0.7;

%[parameters, Commend] = Online_pruning(parameters, Threshold_Encoder_Ratio, Threshold_Decoder_Ratio);

DelayProfiles = ["ETU", "Custom", "EVA", "advanced"]; 

MSE = zeros(3 * Num_of_frame_each_SNR, 1);

index = 0;

for DelayProfile = DelayProfiles

Num_of_QPSK_symbols = Num_of_FFT * Num_of_symbols * Num_of_frame_each_SNR;
Num_of_bits = Num_of_QPSK_symbols * k;

Hybrid_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);

trailingAvg = [];
trailingAvgSq = [];

Denoise_gain_frame = zeros(3 * Num_of_frame_each_SNR, 1);

for Frame = 1 : Num_of_frame_each_SNR

    Training_X = zeros(size(Pilot_location, 1) * size(Pilot_location, 2), 2, 1, batch_size);
    Training_Y = zeros(Num_of_FFT * Num_of_pilot, 2, 1, batch_size);

    MSE_MMSE = zeros(batch_size, 1);
    
    for  i = 1 : batch_size
        
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

        SNR = randi(SNR_Range);
        SNR_OFDM = SNR;
        Doppler_shift = randi([0, MaxDopplerShift]); % MaxDopplerShift
        speedUE = randi([0, floor(50/3.6)]);
        directionUE = -90; % horizontal direction
        t = 0.01; % second, time interval
        numPath = 20;
        velocityUE = speedUE*[cosd(directionUE);sind(directionUE);0]; % 3 x 1

        if contains(DelayProfile, "advanced")
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

            Multitap_Channel_Signal_user_for_channel = Reference_Signal + n;

        else
            [Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_user_for_channel, Reference_Signal, h] = Channel.Propagation_Channel_Model_Online(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, Doppler_shift, DelayProfile);
        end

        % OFDM Receiver
        [Unrecovered_signal, RS_User] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
        [RS, ~] = OFDM.OFDM_Receiver(Multitap_Channel_Signal_user_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Reference_Signal);

        [Received_pilot, ~] = OFDM.Pilot_extract(RS_User, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
        H_Ref = Received_pilot ./ Pilot_value_user;

        % Channel estimation

        % Perfect knowledge on Channel

        % LS
        [Received_pilot_LS, ~] = OFDM.Pilot_extract(Unrecovered_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

        H_LS = CSI.LS(Received_pilot_LS, Pilot_value_user);

        % Pretaining

        %Training_X(:, 1, 1, i) = reshape(real(H_LS), [], 1);
        %Training_X(:, 2, 1, i) = reshape(imag(H_LS), [], 1);

        %Training_Y(:, 1, 1, i) = reshape(real(RS(:, Pilot_location_symbols + 1)), [], 1);
        %Training_Y(:, 2, 1, i) = reshape(imag(RS(:, Pilot_location_symbols + 1)), [], 1);

        %SNR_MMSE = 10 ^ (SNR_OFDM / 10);

        %H_MMSE = CSI.MMSE_Uniform_PDP(length_of_CP, Num_of_FFT, 24, SNR_MMSE, true, RS(:, Pilot_location_symbols + 1));

        %MSE_MMSE = abs(H_MMSE - RS(:, Pilot_location_symbols)) .^ 2;

        %Training_Y(:, 1, 1, i) = reshape(real(H_MMSE), [], 1);
        %Training_Y(:, 2, 1, i) = reshape(imag(H_MMSE), [], 1);

    end

    Reference(:, 1) = RS(Pilot_location(:, 1), 1);
    Reference(:, 2) = RS(Pilot_location(:, 2), 2);

    %MSE_LS = abs(H_LS - Reference) .^ 2;

    %Denoise_gain_frame(end + 1) = 10 * log10(mean(MSE_LS, 'all') ./ mean(MSE_MMSE, 'all'));

        %parameters = pretraining_batch(parameters, Training_X, Training_Y, Commend, batch_size);
        MSE_Hybrid = zeros(Frame_size, 1);

        %[parameters, trailingAvg, trailingAvgSq] = pretraining(parameters, Training_X, Training_Y, Commend, trailingAvg, trailingAvgSq);

for j = 1 : Frame_size

    if contains(DelayProfile, "advanced")
            numPath = 20;
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

            Multitap_Channel_Signal_user_for_channel = Reference_Signal + n;

        else

            [Multitap_Channel_Signal, Multitap_Channel_Signal_user, Multitap_Channel_Signal_user_for_channel, Reference_Signal, h] = Channel.Propagation_Channel_Model_Online(Transmitted_signal, Transmitted_signal_for_channel, SNR_OFDM, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, Doppler_shift, DelayProfile);
    end

    [Unrecovered_signal, ~] = OFDM.OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
    [~, Ref] = OFDM.OFDM_Receiver(Multitap_Channel_Signal_user_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Reference_Signal);

    [Received_pilot_LS, ~] = OFDM.Pilot_extract(Unrecovered_signal, Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

    H_LS = CSI.LS(Received_pilot_LS, Pilot_value_user);

    Feature_signal(:, 1, 1) = reshape(real(H_LS), [], 1);
    Feature_signal(:, 2, 1) = reshape(imag(H_LS), [], 1);
    Feature_signal = dlarray(Feature_signal);

    H_Hybrid_feature = transformer.model(Feature_signal, parameters);

    H_Hybrid_structure = reshape(extractdata(H_Hybrid_feature(:, 1)), Num_of_FFT, Num_of_pilot) + 1j * reshape(extractdata(H_Hybrid_feature(:, 2)), Num_of_FFT, Num_of_pilot);

    H_Hybrid_frame = imresize(H_Hybrid_structure, [Num_of_FFT, max(Pilot_location_symbols)]);
    H_Hybrid_frame(:, max(Pilot_location_symbols) + 1 : Frame_size) = kron(H_Hybrid_frame(:, max(Pilot_location_symbols)), ones(1, Frame_size - max(Pilot_location_symbols)));

    MSE_Hybrid(j, 1) = mean(abs(H_Hybrid_frame - Ref).^2, 'all'); % MSE

    MSE_Hybrid_frame = mean(MSE_Hybrid);

end

Hybrid_MSE_in_frame(Frame, 1) = MSE_Hybrid_frame;

end

index = index + 1;

MSE((index - 1) * Num_of_frame_each_SNR + 1 : index * Num_of_frame_each_SNR) = Hybrid_MSE_in_frame;

end
