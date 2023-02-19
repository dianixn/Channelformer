%% Training options

minibatch_size = 128;

Training_set_ratio = 0.95;
SNR_range = 5:5:25; % 5:5:25
Num_of_frame_each_SNR = 25000;
Channel_type = 'Propogation'; % 'multitaps' 'Rayleigh_doppler' 'Propogation'

numEpochs = 100;
learnRate = 2e-3; % 1.2e-3
Dropperiod = [20:20:100];
Droprate = 0.5;
L2Regularization = 0.0000000001; % 0.0000000001
Validation_frequency = 100;

load_parameters = false;
parameter_file = 'parameters_Transformer';

disp(gpuDeviceTable);

%% Data generation

[Training_X, Training_Y, Validation_X, Validation_Y] = Data_Generation.Data_Generation_Transformer(Training_set_ratio, SNR_range, Num_of_frame_each_SNR, Channel_type);
X = arrayDatastore(reshape(Training_X, size(Training_X, 1), size(Training_X, 2), size(Training_X, 4)), 'IterationDimension', 3);
Y = arrayDatastore(reshape(Training_Y, size(Training_Y, 1), size(Training_Y, 2), size(Training_Y, 4)), 'IterationDimension', 3);
cdsTrain = combine(X, Y);

mbqTrain = minibatchqueue(cdsTrain, 2,...
    'MiniBatchSize', minibatch_size,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat', {'',''},...
    "PartialMiniBatch", "discard");

cdsValidation = combine(arrayDatastore(reshape(Validation_X, size(Training_X, 1), size(Training_X, 2), size(Validation_X, 4)), 'IterationDimension', 3), arrayDatastore(reshape(Validation_Y, size(Validation_Y, 1), size(Validation_Y, 2), size(Validation_Y, 4)), 'IterationDimension', 3));

mbqValidation = minibatchqueue(cdsValidation, 2,...
    'MiniBatchSize', minibatch_size,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat', {'',''},...
    "PartialMiniBatch", "discard");

shuffle(mbqValidation);

Feature_size = size(Training_X, 1);

%% Initialize

if load_parameters == true
    
    load(parameter_file);
    
else
    
    parameters_Transformer.Hyperparameters.NumHeads = 2;
    parameters_Transformer.Hyperparameters.Encoder_num_layers = 1;
    
    for i = 1 : parameters_Transformer.Hyperparameters.Encoder_num_layers
        
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).ln_1_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size); % dlarray(rand(Feature_size, 1) / 1e10);
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).ln_1_b_0 = dlarray(zeros(Feature_size, 1));

        parameters_Transformer.Weights.encoder_layer.("layer_"+i).ln_2_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size);
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).ln_2_b_0 = dlarray(zeros(Feature_size, 1));

        parameters_Transformer.Weights.encoder_layer.("layer_"+i).attn_c_attn_w_0 = initializeGlorot([3 * Feature_size, Feature_size], prod([3 * Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).attn_c_attn_b_0 = dlarray(zeros(3 * Feature_size, 1));

        parameters_Transformer.Weights.encoder_layer.("layer_"+i).attn_c_proj_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).attn_c_proj_b_0 = dlarray(zeros(Feature_size, 1));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).mlp_c_fc_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).mlp_c_fc_b_0 = dlarray(zeros(Feature_size, 1));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).mlp_c_proj_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters_Transformer.Weights.encoder_layer.("layer_"+i).mlp_c_proj_b_0 = dlarray(zeros(Feature_size, 1));
    
    end
    
end

%% Train the model using a custom training loop

% Initialize training progress plot
figure
lineLossTrain = animatedline("Color", [0.8500 0.3250 0.0980]);
lineLossValidation = animatedline("Color", [0 0.4470 0.7410]);

ylim([0 inf]);
xlabel("Iteration");
ylabel("Loss");

% Initialize parameters for the Adam optimizer
trailingAvg = [];
trailingAvgSq = [];

iteration = 0;
start = tic;

% Loop over epochs
for epoch = 1 : numEpochs
    
    % Shuffle data
    shuffle(mbqTrain);
    
    % Loop over mini-batches
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        
        % Read mini-batch of data
        [Training_X_minibatch, Training_Y_minibatch] = next(mbqTrain);
        
        if hasdata(mbqValidation)
            [Xvalidation_minibatch, Yvalidation_minibatch] = next(mbqValidation);
        else
            reset(mbqValidation);
        end
        
        % Evaluate loss and gradients
        [loss, gradients] = dlfeval(@modelGradients, gpuArray(Training_X_minibatch), gpuArray(Training_Y_minibatch), parameters_Transformer);
        
        % Update model parameters
        if ismember(epoch, Dropperiod)
            learnRate = learnRate * Droprate;
        end
        
        for j = 1 : parameters_Transformer.Hyperparameters.Encoder_num_layers
            
            gradients.encoder_layer.("layer_"+j).ln_1_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_1_g_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).ln_1_g_0); % (w * L2Regularization)
            gradients.encoder_layer.("layer_"+j).ln_2_g_0 = dlupdate(@(g, w)  g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).ln_2_g_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).ln_2_g_0);
            gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_attn_w_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).attn_c_attn_w_0);
            gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).attn_c_proj_w_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).attn_c_proj_w_0);
            gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_fc_w_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).mlp_c_fc_w_0);
            gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0 = dlupdate(@(g, w) g + (w * L2Regularization), gradients.encoder_layer.("layer_"+j).mlp_c_proj_w_0, parameters_Transformer.Weights.encoder_layer.("layer_"+j).mlp_c_proj_w_0);
        
        end
        
        [parameters_Transformer.Weights, trailingAvg, trailingAvgSq] = adamupdate(parameters_Transformer.Weights, gradients, ...
            trailingAvg, trailingAvgSq, iteration,learnRate);
        
        % Update training plot
        loss = double(gather(extractdata(loss)));
        addpoints(lineLossTrain, iteration, loss);
        
        if iteration == 1 || mod(iteration, Validation_frequency) == 0
            
            % Validation set
            Prediction_validation = transformer_HA02.model_transformer(Xvalidation_minibatch, parameters_Transformer);
            %loss_validation = Myloss(change_dimension(Yvalidation_minibatch), Prediction_validation) / 100;
            loss_validation = huber(change_dimension(Yvalidation_minibatch), Prediction_validation, "DataFormat", "SCB", 'TransitionPoint', 1);
        
            loss_validation = double(gather(extractdata(loss_validation)));
            addpoints(lineLossValidation, iteration, loss_validation);
            
        end
        
        disp("loss = " + loss)
        disp("Validation loss = " + loss_validation)
        
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

%% Supporting Functions

function [loss, gradients] = modelGradients(X, Y, parameters)

Prediction = transformer_HA02.model_transformer(X, parameters);
loss = huber(change_dimension(Y), Prediction, "DataFormat", "SCB", 'TransitionPoint', 1); % , "DataFormat", "SCB" huber
gradients = dlgradient(loss, parameters.Weights);

end

function [X, Y] = preprocessMiniBatch(XCell, YCell)
    
    % Extract image data from cell and concatenate
    X = cat(4, XCell{:});
    % Extract label data from cell and concatenate
    Y = cat(4, YCell{:});
        
end

function Y = change_dimension(X)
    
    Y = permute(X, [1 2 4 3]);
    
end

function weights = initializeGlorot(sz, numOut, numIn)

Z = 2 * rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end
