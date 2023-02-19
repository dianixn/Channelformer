function Y = Decoder_block(X, parameter)

Weights_1 = parameter.ln_de_w1;
Bias_1 = parameter.ln_de_b1;
Y = dlconv(X, Weights_1, Bias_1, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

Y = relu(Y);

Weights_2 = parameter.ln_de_w2;
Bias_2 = parameter.ln_de_b2;
Y = dlconv(Y, Weights_2, Bias_2, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

Y = Y + X;

Y = transformer.layer.normalization(Y, parameter.ln_de_w3, parameter.ln_de_b3);

end