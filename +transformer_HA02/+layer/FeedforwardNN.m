function Z = FeedforwardNN(X, weights)

Z = transformer_HA02.layer.FC1( X, ...
    weights.mlp_c_fc_w_0, ...
    weights.mlp_c_fc_b_0 );
Z = transformer_HA02.layer.gelu(Z);
Z = transformer_HA02.layer.FC1( Z, ...
    weights.mlp_c_proj_w_0, ...
    weights.mlp_c_proj_b_0 );

end