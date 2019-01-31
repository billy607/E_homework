function r = rownorm(V)
[N,junk] = size(V);
sumd = sum(V);
r = sqrt((sumd-1)*(sumd-1)'/N);
