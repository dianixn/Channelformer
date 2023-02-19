function Z = gelu(X)

Z = 0.5*X.*( 1 + tanh( sqrt(2/pi)*(X+0.044715*(X.^3)) ) );

end