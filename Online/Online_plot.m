%% Figure
%space = 50;

for i = 1 : (40000 / 10000)
    MSE_frame = MSE_no_pruning((i - 1) * 10000 + 1 : 10000 : i * 10000);
    MSE_space_no_prune(i, 1) = mean(MSE_frame);
end

figure;
semilogy(1 : 1 : size(MSE, 1), kron(MSE_space_no_prune, ones(10000, 1)), 'LineWidth', 1);
hold on;
semilogy(1 : space : size(MSE, 1), MSE_space_5dB, 'LineWidth', 1);
hold on;
semilogy(1 : space : size(MSE, 1), MSE_space_MMSE, 'LineWidth', 1);
hold on;
semilogy(1 : space : size(MSE, 1), MSE_space_5dB_pruning, 'LineWidth', 1);
hold on;
semilogy(1 : space : size(MSE, 1), MSE_space_MMSE_pruning, 'LineWidth', 1);

ylim([5e-4 5e1])

legend('Online training not applied', ...
    'Online training applied (LS 5dB)', ...
    'Online training applied (MMSE)', ...
    'Online training applied for 70% weight-level pruning (LS 5dB)', ...
    'Online training applied for 70% weight-level pruning (MMSE)');

xlabel('Channel realizations');
ylabel('MSE');
title('Dynamic online test on the ETU, customized, EVA and advanced channels');
grid on;
hold off;