function [Z, present] = Encoder_block(X, weights, hyperParameters)

[A, present] = transformer.layer.attention(X, [], weights, hyperParameters); 

A = A + permute(X, [1 2 4 3]); 

A = transformer.layer.normalization(A, ...
    weights.ln_1_g_0, weights.ln_1_b_0);

A = permute(A, [1 2 4 3]);

Z = transformer.layer.FeedforwardNN(A, weights);

Z = Z + A;

Z = transformer.layer.normalization(Z, ...
    weights.ln_2_g_0, weights.ln_2_b_0);

end