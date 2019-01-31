function V = rowcol(V)
[N,junk] = size(V);
sumc = sum(V);
sumc = diag(sumc);
sumc = ones(N)*sumc;
V = V ./sumc;
sumr = sum(V');
sumr = diag(sumr);
sumr = ones(N)*sumr;
V = V ./sumr';

