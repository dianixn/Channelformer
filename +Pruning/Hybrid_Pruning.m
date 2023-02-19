function [parameters, Commend] = Hybrid_Pruning(File, Threshold_Encoder_Ratio, Threshold_Decoder_Ratio)

%File = 'parameters_HA03';
%Threshold_Encoder_Ratio = 0.5;
%Threshold_Decoder_Ratio = 0.5;

% Encoder and Decoder Pruning

load(File);

parameters = parameters_frame;

Commend = "";

if Threshold_Encoder_Ratio ~= 0
    Encoder_pruning = true;
else
    Encoder_pruning = false;
end

if Threshold_Decoder_Ratio ~= 0
    Decoder_pruning = true;
else
    Decoder_pruning = false;
end

Weight_Encoder = [];
Weight_Decoder = [];

if Encoder_pruning == true
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                Weight_Encoder = [Weight_Encoder; reshape(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1}), [], 1)];
            end
    
        end
        
    end
    
    Weight_value_Encoder = sort(abs(extractdata(Weight_Encoder)));

    semilogx(Weight_value_Encoder, (0 : size(Weight_value_Encoder, 1) - 1) / (size(Weight_value_Encoder, 1) - 1));

    xlabel('Weight Value at the encoder');
    ylabel('Probability');

    Threshold_Value_Transformer = Weight_value_Encoder(fix(Threshold_Encoder_Ratio * size(Weight_value_Encoder, 1)));
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})(abs(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})) < Threshold_Value_Transformer) = 0;
                Pruning_index_Encoder = gather(find((extractdata(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1}) == 0)))); % need 
                
                for k = 1 : length(Pruning_index_Encoder)
                    commend = "gradients" + '.' + "encoder_layer" + '.' + ("layer_"+i) + '.' + (Layer_Name{j, 1}) + "(" + num2str(Pruning_index_Encoder(k)) + ")" + ' = 0;'; % need
                    Commend = Commend + commend;
                end
                
            end
    
        end
        
    end
    
end

if Decoder_pruning == true
    
    for i = [1 : parameters.Hyperparameters.Decoder_num_layers, 31] %31
        
        Layer_Name = fieldnames(parameters.Weights.decoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w")
                Weight_Decoder = [Weight_Decoder; reshape(parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1}), [], 1)];
            end
    
        end
        
    end
    
    Weight_value_Decoder = sort(abs(extractdata(Weight_Decoder)));

    semilogx(Weight_value_Decoder, (0 : size(Weight_value_Decoder, 1) - 1) / (size(Weight_value_Decoder, 1) - 1));

    xlabel('Weight Value at the decoder');
    ylabel('Probability');

    Threshold_Value_decoder = Weight_value_Decoder(fix(Threshold_Decoder_Ratio * size(Weight_value_Decoder, 1)));
    
    for i = [1 : parameters.Hyperparameters.Decoder_num_layers, 31] %31
        
        Layer_Name = fieldnames(parameters.Weights.decoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)
            
            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w")
                parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1})(abs(parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1})) < Threshold_Value_decoder) = 0;
                Pruning_index_Decoder = gather(find((extractdata(parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1}) == 0))));
                
                for k = 1 : length(Pruning_index_Decoder)
                    commend = "gradients" + '.' + "decoder_layer" + '.' + ("layer_"+i) + '.' + (Layer_Name{j, 1}) + "(" + num2str(Pruning_index_Decoder(k)) + ")" + ' = 0;'; % need 
                    Commend = Commend + commend;
                end
                
            end
    
        end
        
    end
    
end
