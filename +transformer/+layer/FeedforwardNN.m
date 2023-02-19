function Z = FeedforwardNN(X, weights)

Z = dlconv(X, weights.mlp_c_fc_w_0, weights.mlp_c_fc_b_0, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

Z = transformer.layer.gelu(Z);

Z = dlconv(Z, weights.mlp_c_proj_w_0, weights.mlp_c_proj_b_0, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

end