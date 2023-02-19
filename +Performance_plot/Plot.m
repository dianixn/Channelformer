figure;
semilogy(SNR_Range,MSE_LS_over_SNR, 'Marker', '*', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_DDCE_over_SNR, 'Marker', '*', 'LineWidth', 1, 'LineStyle', ':');
hold on
semilogy(SNR_Range,MSE_MMSE_over_SNR, 'Marker', 'o', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_MME_over_SNR, 'Marker', 'o', 'LineWidth', 1, 'LineStyle', '--');
hold on
semilogy(SNR_Range,MSE_DNN_over_SNR, 'Marker', '+', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_ResNet_over_SNR, 'Marker', 'x', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_Transformer_over_SNR, 'Marker', 's', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_HA02_over_SNR, 'Marker', 'v', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_Hybrid_over_SNR, 'Marker', 'd', 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_Hybrid_frame_over_SNR, 'Marker', 'h', 'LineWidth', 1);

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

xlabel('SNR in dB');
ylabel('MSE');
title('MSE Performance over SNR on the ETU channel');
grid on;
hold off;

figure;
semilogy(0:5:30,MSE_LS_over_SNR, 'Marker', '*', 'LineWidth', 1);
hold on
semilogy(0:5:30,MSE_DDCE_over_SNR, 'Marker', '*', 'LineWidth', 1, 'LineStyle', ':');
legend('LS', ...
    'DD-CE'); 
xlabel('SNR in dB');
ylabel('MSE');
title('MSE Performance');
grid on;
hold off;