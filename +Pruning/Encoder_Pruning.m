% Encoder Pruning

load('parameters_commRayleygh_ETU_Transformer.mat');

Threshold_Encoder_Ratio = 0.1;

Weight = [];
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                Weight = [Weight; reshape(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1}), [], 1)];
            end
    
        end
        
    end
    
    Weight_value = sort(abs(extractdata(Weight)));

    semilogx(Weight_value, (0 : size(Weight_value, 1) - 1) / (size(Weight_value, 1) - 1));

    xlabel('Weight Value');
    ylabel('Probability');

    Threshold_Value = Weight(fix(Threshold_Encoder_Ratio * size(Weight_value, 1)));
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})(abs(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})) < Threshold_Value) = 0;
            end
    
        end
        
    end
