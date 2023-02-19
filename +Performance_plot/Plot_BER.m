figure;
semilogy(SNR_Range,BER_LS_over_SNR, 'Marker', '*', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_DDCE_over_SNR, 'Marker', '*', 'LineWidth', 1, 'LineStyle', ':');
hold on
semilogy(SNR_Range,BER_MMSE_over_SNR, 'Marker', 'o', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_MME_over_SNR, 'Marker', 'o', 'LineWidth', 1, 'LineStyle', '--');
hold on
semilogy(SNR_Range,BER_DNN_over_SNR, 'Marker', '+', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_ResNet_over_SNR, 'Marker', 'x', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_Transformer_over_SNR, 'Marker', 's', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_HA02_over_SNR, 'Marker', 'v', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_Hybrid_over_SNR, 'Marker', 'd', 'LineWidth', 1);
hold on
semilogy(SNR_Range,BER_Hybrid_frame_over_SNR, 'Marker', 'h', 'LineWidth', 1);

ylim([1e-4 1])

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
ylabel('BER');
title('BER Performance over SNR on the ETU channel');
%, 'fontweight','bold'
grid on;
hold off;
