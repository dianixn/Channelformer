figure;
semilogy(DopplerShift, MSE_Hybrid_over_SNR_full, 'Marker', '*', 'LineWidth', 1, 'color', [0 0.4470 0.7410]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_10, 'Marker', 'o', 'LineWidth', 1, 'color', [0.8500 0.3250 0.0980]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_30, 'Marker', '+', 'LineWidth', 1, 'LineStyle', ':', 'color', [0.4660 0.6740 0.1880]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_50, 'Marker', 'x', 'LineWidth', 1, 'LineStyle', '--', 'color', [0.6350 0.0780 0.1840]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_70, 'Marker', 's', 'LineWidth', 1, 'color', [0.4940 0.1840 0.5560]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_80, 'Marker', 'v', 'LineWidth', 1, 'LineStyle', '-.', 'color', [0.9290 0.6940 0.1250]);
hold on
semilogy(DopplerShift, MSE_Hybrid_over_SNR_90, 'Marker', 'd', 'LineWidth', 1, 'color', [0.3010 0.7450 0.9330]);

ylim([1e-3 1e2])

legend('Online Channelformer', ...
    'Online Channelformer with 10% customized weight-level pruning', ...
    'Online Channelformer with 30% customized weight-level pruning', ...
    'Online Channelformer with 50% customized weight-level pruning', ...
    'Online Channelformer with 70% customized weight-level pruning', ...
    'Online Channelformer with 80% customized weight-level pruning', ...
    'Online Channelformer with 90% customized weight-level pruning');

xlabel('Doppler shift in Hz');
ylabel('MSE');
title('MSE performance over the extended Doppler shift');
grid on;
hold off;