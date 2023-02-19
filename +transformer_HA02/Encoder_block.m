function [Z, present] = Encoder_block(X, weights, hyperParameters)

[A, present] = transformer_HA02.layer.attention(X, [], weights, hyperParameters); 

A = A + X; 

A = transformer_HA02.layer.normalization(A, ...
    weights.ln_1_g_0, weights.ln_1_b_0);

Z = transformer_HA02.layer.FeedforwardNN(A, weights);

Z = Z + A;

Z = transformer_HA02.layer.normalization(Z, ...
    weights.ln_2_g_0, weights.ln_2_b_0);

end