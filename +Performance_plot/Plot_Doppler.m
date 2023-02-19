figure;
semilogy(DopplerShift,MSE_LS_over_Doppler, 'Marker', '*', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_DDCE_over_Doppler, 'Marker', '*', 'LineWidth', 1, 'LineStyle', ':');
hold on
semilogy(DopplerShift,MSE_MMSE_over_Doppler, 'Marker', 'o', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_MME_over_Doppler, 'Marker', 'o', 'LineWidth', 1, 'LineStyle', '--');
hold on
semilogy(DopplerShift,MSE_DNN_over_Doppler, 'Marker', '+', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_ResNet_over_Doppler, 'Marker', 'x', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_Transformer_over_Doppler, 'Marker', 's', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_HA02_over_Doppler, 'Marker', 'v', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_Hybrid_over_Doppler, 'Marker', 'd', 'LineWidth', 1);
hold on
semilogy(DopplerShift,MSE_Hybrid_frame_over_Doppler, 'Marker', 'h', 'LineWidth', 1);

ylim([1e-6 10])

legend('LS', ...
    'DD-CE', ...
    '1D FD-MMSE', ...
    '2D FD-MMSE', ...
    'Interpolation-ResNet', ...
    'ReEsNet', ...
    'TR', ...
    'HA02', ...
    'Online Channelformer', ...
    'Offline Channelformer'); 

xlabel('Doppler Shift in Hz');
ylabel('MSE');
title('MSE performance over Doppler Shift on the ETU channel');
grid on;
hold off;
