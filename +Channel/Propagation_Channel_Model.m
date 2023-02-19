function [Fading_signal, Fading_signal_noise_free, Reference_Signal] = Propagation_Channel_Model(Transmitted_signal, Transmitted_signal_for_channel, SNR, SampleRate, Carrier_Frequency, PathDelays, AveragePathGains, MaxDopplerShift, DelayProfile)

channel.NRxAnts = 1;
channel.MIMOCorrelation = 'Low';
channel.DelayProfile = DelayProfile;
channel.NormalizeTxAnts = 'On';

channel.DopplerFreq = MaxDopplerShift;
channel.CarrierFreq = Carrier_Frequency;
channel.SamplingRate = SampleRate;
channel.InitTime = 0;
channel.InitPhase = 'Random';
channel.Seed = randi([0, 1e9]);
channel.ModelType = 'GMEDS';
channel.NTerms = 16;
channel.NormalizePathGains = 'On';

if contains(channel.DelayProfile,'Custom')
    channel.AveragePathGaindB = AveragePathGains;
    channel.PathDelays = PathDelays; 
end

[Fading_signal_noise_free, ~] = lteFadingChannel(channel, Transmitted_signal);
[Reference_Signal, ~] = lteFadingChannel(channel, Transmitted_signal_for_channel);
%[h, ~] = lteFadingChannel(channel, [1; zeros(size(Transmitted_signal_for_channel, 1) - 1, 1)]);

% Noise Gneration
SignalPower = mean(abs(Fading_signal_noise_free) .^ 2);
Noise_Variance = SignalPower / (10 ^ (SNR / 10));

Nvariance = sqrt(Noise_Variance / 2);
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

Fading_signal = Fading_signal_noise_free + n;
