% Parameters

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 71; %126
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16 + 7;

Num_of_symbols = 12;
Num_of_pilot = 2;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_location_symbols = [1, 13];
Pilot_location = [(1 : 2 : Num_of_FFT)', (2 : 2 : Num_of_FFT)'];
Pilot_value_user = 1 + 1j;

length_of_symbol = Num_of_FFT + length_of_CP;

SNR = 10;

Frequency_Spacing = 15e3;

Carrier_Frequency = 2.1e9;
Max_Mobile_Speed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; % km/h

SampleRate = Num_of_subcarriers * Frequency_Spacing;

PathDelays = [0 50 120 200 230 500 1600 2300 5000 7000] * 1e-9; 
AveragePathGains = [-1.0 -1.0 -1.0 -1.0 -1.0 -1.5 -1.5 -1.5 -3.0 -5.0]; 

DopplerShift = floor((Carrier_Frequency * Max_Mobile_Speed) / (3e8 * 3.6));

DelayProfile = 'ETU'; % 'EPA' 'EVA' 'ETU' 'Custom'
