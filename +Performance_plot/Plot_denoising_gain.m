figure;
plot(SNR_Range, MSE_Hybrid_over_SNR_full_Denoise, 'Marker', '*', 'LineWidth', 1, 'color', [0 0.4470 0.7410]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_10_Denoise, 'Marker', 'o', 'LineWidth', 1, 'color', [0.8500 0.3250 0.0980]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_30_Denoise, 'Marker', '+', 'LineWidth', 1, 'LineStyle', ':', 'color', [0.4660 0.6740 0.1880]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_50_Denoise, 'Marker', 'x', 'LineWidth', 1, 'LineStyle', '--', 'color', [0.6350 0.0780 0.1840]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_70_Denoise, 'Marker', 's', 'LineWidth', 1, 'color', [0.4940 0.1840 0.5560]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_80_Denoise, 'Marker', 'v', 'LineWidth', 1, 'LineStyle', '-.', 'color', [0.9290 0.6940 0.1250]);
hold on
plot(SNR_Range, MSE_Hybrid_over_SNR_90_Denoise, 'Marker', 'd', 'LineWidth', 1, 'color', [0.3010 0.7450 0.9330]);

ylim([-10 30])

legend('Online Channelformer', ...
    'Online Channelformer with 10% weight-level pruning without fine-tuning', ...
    'Online Channelformer with 30% weight-level pruning without fine-tuning', ...
    'Online Channelformer with 50% weight-level pruning without fine-tuning', ...
    'Online Channelformer with 70% weight-level pruning without fine-tuning', ...
    'Online Channelformer with 80% weight-level pruning without fine-tuning', ...
    'Online Channelformer with 90% weight-level pruning without fine-tuning');

xlabel('SNR in dB');
ylabel('Denoising gain in dB');
title('Denoising gain over the extended SNR range');
grid on;
hold off;