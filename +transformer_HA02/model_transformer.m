function Z = model_transformer(x, parameters)

w = parameters.Weights;
hyperparameters = parameters.Hyperparameters;

Weights = w.decoder_layer.ln_de_w1;
Bias = w.decoder_layer.ln_de_b1;
Z = dlconv(x, Weights, Bias, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

Z = permute(Z, [1 2 4 3]);

% Transformer

for i = 1 : hyperparameters.Encoder_num_layers
    Z = transformer_HA02.Encoder_block(Z, w.encoder_layer.("layer_"+i), hyperparameters);
end

end
