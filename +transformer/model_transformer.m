function Z = model_transformer(x, parameters)

w = parameters.Weights;
hyperparameters = parameters.Hyperparameters;

Z = x;

% Transformer

for i = 1 : hyperparameters.Encoder_num_layers
    Z = transformer.Encoder_block_transformer(Z, w.encoder_layer.("layer_"+i), hyperparameters);
end

Z = permute(Z, [1 2 4 3]);

end
