function parameters = pretraining_batch(parameters, Training_X, Training_Y, Commend, minibatch_size)

learnRate = 1e-3;
L2Regularization = 0.0000001;

%% Train the model using a custom training loop

X = arrayDatastore(reshape(Training_X, size(Training_X, 1), size(Training_X, 2), size(Training_X, 4)), 'IterationDimension', 3);
Y = arrayDatastore(reshape(Training_Y, size(Training_Y, 1), size(Training_Y, 2), size(Training_Y, 4)), 'IterationDimension', 3);
cdsTrain = combine(X, Y);

mbqTrain = minibatchqueue(cdsTrain, 2,...
    'MiniBatchSize', minibatch_size,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat', {'',''},...
    "PartialMiniBatch", "discard");

% Initialize parameters for the Adam optimizer
trailingAvg = [];
trailingAvgSq = [];

iteration = 0;

while hasdata(mbqTrain)
        iteration = iteration + 1;
        
        % Read mini-batch of data
        [Training_X_minibatch, Training_Y_minibatch] = next(mbqTrain);
        
        % Evaluate loss and gradients
        [~, gradients] = dlfeval(@modelGradients, gpuArray(Training_X_minibatch), gpuArray(Training_Y_minibatch), parameters);
        
        
        for j = 1 : parameters.Hyperparameters.Encoder_num_layers
            
            gradients.encoder_layer.("layer_"+j).ln_1_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_1_g_0, parameters.Weights.encoder_layer.("layer_"+j).ln_1_g_0); % (w * L2Regularization)
            gradients.encoder_layer.("layer_"+j).ln_2_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_2_g_0, parameters.Weights.encoder_layer.("layer_"+j).ln_2_g_0);
            gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0, parameters.Weights.encoder_layer.("layer_"+j).attn_c_attn_w_0);
            gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0, parameters.Weights.encoder_layer.("layer_"+j).attn_c_proj_w_0);
            gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0, parameters.Weights.encoder_layer.("layer_"+j).mlp_c_fc_w_0);
            gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0, parameters.Weights.encoder_layer.("layer_"+j).mlp_c_proj_w_0);
        
        end
        
        for j = 1 : parameters.Hyperparameters.Decoder_num_layers
            
            gradients.decoder_layer.("layer_"+j).ln_de_w1 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w1, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w1); % (w * L2Regularization)
            gradients.decoder_layer.("layer_"+j).ln_de_w2 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w2, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w2);
            gradients.decoder_layer.("layer_"+j).ln_de_w3 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.decoder_layer.("layer_"+j).ln_de_w3, parameters.Weights.decoder_layer.("layer_"+j).ln_de_w3);
            
        end
        
        gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1, parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1);
        
        eval(Commend);
        
        [parameters.Weights, trailingAvg, trailingAvgSq] = adamupdate(parameters.Weights, gradients, ...
            trailingAvg, trailingAvgSq, iteration,learnRate);

end
        
end

%% Functions

function [loss, gradients] = modelGradients(X, Y, parameters)

Prediction = transformer.model(X, parameters);
loss = huber(Y, Prediction, "DataFormat", "SSCB", 'TransitionPoint', 1); % , "DataFormat", "SCB" huber change_dimension(Y)
gradients = dlgradient(loss, parameters.Weights);

end

function [X, Y] = preprocessMiniBatch(XCell, YCell)
    
    % Extract image data from cell and concatenate
    X = cat(4, XCell{:});
    % Extract label data from cell and concatenate
    Y = cat(4, YCell{:});
        
end
