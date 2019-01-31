function p = permnorm(V)
[N,junk] = size(V);
p = sqrt(trace((V'*V-eye(N))'*(V'*V-eye(N)))/N);
