%% Figure
space = 100;

load('MSE_space_LS_5dB');
load('MSE_space_MMSE');
load('MSE_space_MMSE_trained');
load('MSE_space_trained');
load('MSE_space_untrained');

figure;
plot(1 : 1 : 15000, MSE_space_untrained_test, 'color', [0.8500 0.3250 0.0980]);
hold on;
plot(1 : space : space * size(MSE_space_MMSE, 1), MSE_space_LS_5dB, 'color', [0 0.4470 0.7410]);
hold on;
plot(1 : space : space * size(MSE_space_MMSE_trained, 1), MSE_space_trained, 'color', [0.4660 0.6740 0.1880]);
hold on;
plot(1 : space : space * size(MSE_space_trained, 1), MSE_space_MMSE, 'color', [0.6350 0.0780 0.1840]);
hold on;
plot(1 : space : space * size(MSE_space_untrained, 1), MSE_space_MMSE_trained, 'color', [0.4940 0.1840 0.5560]);

ylim([0 0.2])
%ylim([0 5])

legend('Offline training only', ...
    'Online training applied (LS 5dB)', ...
    'Online training applied for 70% customized pruning (LS 5dB)', ...
    'Online training applied (MMSE)', ...
    'Online training applied for 70% customized pruning (MMSE)');

xlabel('Channel realizations');
ylabel('MSE');
title('Dynamic online test on the 3GPP channels');
grid on;
hold off;
