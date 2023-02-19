function Z = model(x, parameters)

w = parameters.Weights;
hyperparameters = parameters.Hyperparameters;

Z = x;

% Transformer layers

for i = 1 : hyperparameters.Encoder_num_layers
    Z = transformer.HA03.Encoder_block(Z, w.encoder_layer.("layer_"+i), hyperparameters);
end

% Regression

Weights = w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_w;
Bias = w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_b;
Z = dlconv(Z, Weights, Bias, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

for j = 1 : hyperparameters.Decoder_num_layers
    Z = transformer.HA03.Decoder_block(Z, w.decoder_layer.("layer_"+j));
end

Z = transformer.layer.FC1(Z, w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_w1, w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_b1);

Weights = w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_w0;
Bias = w.decoder_layer.("layer_" + hyperparameters.Decoder_num_layers + 1).ln_de_b0;
Z = dlconv(Z, Weights, Bias, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

end
