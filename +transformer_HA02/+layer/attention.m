function [A, present] = attention(X, past, weights, hyperParameters, nvp)

% Use a fully connected layer to generate queries, keys and values from the
% input.
C = transformer_HA02.layer.FC1(X, ...
    weights.attn_c_attn_w_0, ...
    weights.attn_c_attn_b_0 );

% Split the results into Q (Query), K (Keys) and V (Values).
splitSize = size(C,1)/3;
Q = C(1:splitSize,:,:);
K = C((splitSize+1):(2*splitSize),:,:);
V = C((2*splitSize+1):(3*splitSize),:,:);

% Split heads
Q = iSplitHeads(Q, splitSize, hyperParameters.NumHeads);
K = iSplitHeads(K, splitSize, hyperParameters.NumHeads);
V = iSplitHeads(V, splitSize, hyperParameters.NumHeads);

% Use the past
if ~isempty(past)
    PK = past(:,:,:,:,1);
    PV = past(:,:,:,:,2);
    K = cat(2,PK,K);
    V = cat(2,PV,V);
end

present = cat(5,K,V);

A = transformer_HA02.layer.multiheadAttention(Q,K,V);

A = iMergeHeads(A); % A (numFeatures*numHeads)-by-numSubwords-by-numObs array.

A = transformer_HA02.layer.FC1( A, ...
    weights.attn_c_proj_w_0, ...
    weights.attn_c_proj_b_0 );
end

function Z = iSplitHeads(X, splitSize, numHeads)

X = reshape(X, splitSize/numHeads, numHeads, [], size(X,3));
Z = permute(X,[1 3 2 4]);
end

function Z = iMergeHeads(X)

X = permute(X, [1 3 2 4]);
Z = reshape(X, size(X,1)*size(X,2), [], size(X,4));

end
