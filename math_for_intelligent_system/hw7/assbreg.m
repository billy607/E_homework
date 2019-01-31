function V = assbreg(A)
[N,junk] = size(A);
W = ones(N)/N;
V = W;
alpha = 1;
l2thr = 0.01;
permthr = 0.1;
p = permnorm(V);
energy = trace(A'*V);
ergplot = energy;
permplot = p;
while p > permthr
	V = W.*exp(alpha*A);
        sinkhorn = 0;
	while rownorm(V) > l2thr,
		V = rowcol(V);
                sinkhorn = sinkhorn + 1;
	end
  W = V;
	energy = trace(A'*V)	;	
	p = permnorm(V);
  ergplot = [ergplot,energy];
  permplot = [permplot, p];
end
figure(1)
plot(ergplot,'linewidth',3)
xlabel('iterations','fontsize',14);
ylabel("trace(A^T V)",'Interpreter','tex','fontsize',14);
figure(2)
plot(permplot,'linewidth',3)
xlabel('iterations','fontsize',14);
ylabel('permutation norm','fontsize',14);


