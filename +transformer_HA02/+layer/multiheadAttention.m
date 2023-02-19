function A = multiheadAttention(Q, K, V)

W = dlmtimes(permute(K, [2 1 3 4]), Q);

% Divide by square root of d
W = W./sqrt(size(Q,1));

% Apply softmax
W = softmax(W, 'DataFormat', 'CTUB');

A = dlmtimes(V, W);

end
