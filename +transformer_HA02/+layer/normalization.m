function Z = normalization(X, g, b)

normalizationDimension = 1;

epsilon = single(1e-5);

U = mean(X, normalizationDimension);
S = mean((X-U).^2, normalizationDimension);
X = (X-U) ./ sqrt(S + epsilon);
Z = g.*X + b;

end