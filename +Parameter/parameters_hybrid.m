%% Encoder

for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        parameters.Weights.encoder_layer.("layer_"+i).ln_1_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size); % dlarray(rand(Feature_size, 1) / 1e10);
        parameters.Weights.encoder_layer.("layer_"+i).ln_1_b_0 = dlarray(zeros(Feature_size, 1));

        parameters.Weights.encoder_layer.("layer_"+i).ln_2_g_0 = initializeGlorot([Feature_size, 1], Feature_size, Feature_size);
        parameters.Weights.encoder_layer.("layer_"+i).ln_2_b_0 = dlarray(zeros(Feature_size, 1));

        parameters.Weights.encoder_layer.("layer_"+i).attn_c_attn_w_0 = initializeGlorot([3 * Feature_size, Feature_size], prod([3 * Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters.Weights.encoder_layer.("layer_"+i).attn_c_attn_b_0 = dlarray(zeros(3 * Feature_size, 1));

        parameters.Weights.encoder_layer.("layer_"+i).attn_c_proj_w_0 = initializeGlorot([Feature_size, Feature_size], prod([Feature_size, Feature_size]), prod([Feature_size, Feature_size]));
        parameters.Weights.encoder_layer.("layer_"+i).attn_c_proj_b_0 = dlarray(zeros(Feature_size, 1));
        
        Number_of_filters_Encoder = 5;

        filterSize = [3 3];
        numChannels = 1;
        numFilters = Number_of_filters_Encoder;

        sz = [filterSize numChannels numFilters];
        numOut = prod(filterSize) * numFilters;
        numIn = prod(filterSize) * numFilters;

        parameters.Weights.encoder_layer.("layer_"+i).mlp_c_fc_w_0 = initializeGlorot(sz, numOut, numIn);
        parameters.Weights.encoder_layer.("layer_"+i).mlp_c_fc_b_0 = dlarray(zeros(numFilters, 1));

        filterSize = [3 3];
        numChannels = Number_of_filters_Encoder;
        numFilters = 1;

        sz = [filterSize numChannels numFilters];
        numOut = prod(filterSize) * numFilters;
        numIn = prod(filterSize) * numFilters;

        parameters.Weights.encoder_layer.("layer_"+i).mlp_c_proj_w_0 = initializeGlorot(sz, numOut, numIn);
        parameters.Weights.encoder_layer.("layer_"+i).mlp_c_proj_b_0 = dlarray(zeros(numFilters, 1));
    
end

%% Decoder

Number_of_filters = 12;

filterSize = [5 5];
numChannels = 1;
numFilters = Number_of_filters;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w = initializeGlorot(sz, numOut, numIn);
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b = dlarray(zeros(numFilters, 1));

for j = 1 : parameters.Hyperparameters.Decoder_num_layers
    
    filterSize = [5 5];
    numChannels = Number_of_filters;
    numFilters = Number_of_filters;

    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    parameters.Weights.decoder_layer.("layer_" + j).ln_de_w1 = initializeGlorot(sz, numOut, numIn);
    parameters.Weights.decoder_layer.("layer_" + j).ln_de_b1 = dlarray(zeros(numFilters, 1));
    
    filterSize = [5 5];
    numChannels = Number_of_filters;
    numFilters = Number_of_filters;

    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    parameters.Weights.decoder_layer.("layer_" + j).ln_de_w2 = initializeGlorot(sz, numOut, numIn);
    parameters.Weights.decoder_layer.("layer_" + j).ln_de_b2 = dlarray(zeros(numFilters, 1));
    
    parameters.Weights.decoder_layer.("layer_"+j).ln_de_w3 = initializeGlorot([Feature_size, 1], prod([Feature_size, 1]), prod([Feature_size, 1]));
    parameters.Weights.decoder_layer.("layer_"+j).ln_de_b3 = dlarray(zeros(Feature_size, 1));
    
end

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1 = initializeGlorot([size(Training_Y, 1), Feature_size], prod([size(Training_Y, 1), Feature_size]), prod([size(Training_Y, 1), Feature_size]));
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b1 = dlarray(zeros(size(Training_Y, 1), 1));

filterSize = [5 5];
numChannels = Number_of_filters;
numFilters = 1;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w0 = initializeGlorot(sz, numOut, numIn);
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b0 = dlarray(zeros(numFilters, 1));

%% initializeGlorot

function weights = initializeGlorot(sz, numOut, numIn)

Z = 2 * rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end
