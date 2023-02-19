load('Attention_ETU');
load('Attention_EVA');
load('Attention_EPA');
load('Attention_Custom');

head = 2;
Channel = 2;
figure;

Plot = 0;

if Plot == 1
    
    Attention_mean_ETU = 0;

    for i = 1 : size(Attention, 2)
        
        Attention_head = extractdata(Attention{1, i});
        Attention_Channel = abs(Attention_head(:, 1, head));
        Attention_mean_ETU = Attention_mean_ETU + Attention_Channel;
    
    end
    
    Attention_mean_ETU = Attention_mean_ETU / size(Attention, 2);
    
    plot(Attention_mean_ETU, 'color', [0.8500 0.3250 0.0980]);
    hold on

    for i = 1 : size(Attention, 2)
        
        Attention_head = extractdata(Attention{1, i});
        Attention_Channel = Attention_head(:, Channel, head);
        plot(abs(Attention_Channel), '.', 'color', [0 0.4470 0.7410]);
        hold on
        
    end
    
    ylim([0 10])
    
    legend('Mean', ...
    'Distribution of the magnitude of value V');
    
    xlabel('Index of value V');
    ylabel('Magnitude');
    title('Attention of head' + string(head) + 'and channel' + string(Channel));
    grid on;
    hold off;

elseif Plot == 2
    
    Attention_mean_ETU = 0;
    Attention_mean_EVA = 0;

    for i = 1 : size(Attention, 2)
        
        Attention_head = extractdata(Attention{1, i});
        Attention_Channel = abs(Attention_head(:, 1, head));
        Attention_mean_ETU = Attention_mean_ETU + Attention_Channel;
    
    end
    
    for i = 1 : size(Attention, 2)
        
        Attention_head = extractdata(Attention{1, i});
        Attention_Channel = abs(Attention_head(:, 2, head));
        Attention_mean_EVA = Attention_mean_EVA + Attention_Channel;
    
    end

    Attention_mean_ETU = Attention_mean_ETU / size(Attention, 2);
    
    Attention_mean_EVA = Attention_mean_EVA / size(Attention, 2);
    
    plot(Attention_mean_ETU, 'color', [0 0.4470 0.7410]);
    hold on
    plot(Attention_mean_EVA, 'color', [0.8500 0.3250 0.0980]);
    
    ylim([0.39 1.25])
    
    legend('Channel 1', ...
    'Channel 2');

    xlabel('Index of value');
    ylabel('Magnitude');
    title('Scaled dot-product attention of head' + string(head));
    grid on;
    hold off;

else

    Attention_mean_ETU = 0;
    Attention_mean_EVA = 0;
    Attention_mean_EPA = 0;
    Attention_mean_Custom = 0;

    for i = 1 : size(Attention_ETU, 2)
        
        Attention_head = extractdata(Attention_ETU{1, i});
        Attention_Channel_ETU = abs(Attention_head(:, Channel, head));
        Attention_mean_ETU = Attention_mean_ETU + Attention_Channel_ETU;
    
    end
    
    for i = 1 : size(Attention_EVA, 2)
        
        Attention_head = extractdata(Attention_EVA{1, i});
        Attention_Channel_EVA = abs(Attention_head(:, Channel, head));
        Attention_mean_EVA = Attention_mean_EVA + Attention_Channel_EVA;
    
    end

    for i = 1 : size(Attention_EPA, 2)
        
        Attention_head = extractdata(Attention_EPA{1, i});
        Attention_Channel_EPA = abs(Attention_head(:, Channel, head));
        Attention_mean_EPA = Attention_mean_EPA + Attention_Channel_EPA;
    
    end

    for i = 1 : size(Attention_Custom, 2)
        
        Attention_head = extractdata(Attention_Custom{1, i});
        Attention_Channel_Custom = abs(Attention_head(:, Channel, head));
        Attention_mean_Custom = Attention_mean_Custom + Attention_Channel_Custom;
    
    end

    Attention_mean_ETU = Attention_mean_ETU / size(Attention_ETU, 2);
    
    Attention_mean_EVA = Attention_mean_EVA / size(Attention_EVA, 2);

    Attention_mean_EPA = Attention_mean_EPA / size(Attention_EPA, 2);

    Attention_mean_Custom = Attention_mean_Custom / size(Attention_Custom, 2);
    
    semilogy(Attention_mean_EPA, 'Marker', '*', 'LineWidth', 1, 'color', [0 0.4470 0.7410]);
    hold on
    semilogy(Attention_mean_EVA, 'Marker', 'o', 'LineWidth', 1, 'color', [0.8500 0.3250 0.0980]);
    hold on
    semilogy(Attention_mean_ETU, 'Marker', '+', 'LineWidth', 1, 'LineStyle', ':', 'color', [0.4660 0.6740 0.1880]);
    hold on
    semilogy(Attention_mean_Custom, 'Marker', 'x', 'LineWidth', 1, 'LineStyle', '--', 'color', [0.6350 0.0780 0.1840]);
    
    ylim([1e-1 1])
    
    legend('EPA', ...
    'EVA', ...
    'ETU', ...
    'Customized');

    xlabel('Index of elements');
    ylabel('Magnitude');
    title('Scaled dot-product attention of head' + string(head) + 'and channel' + string(Channel), 'fontweight','bold');
    grid on;
    hold off;

end
