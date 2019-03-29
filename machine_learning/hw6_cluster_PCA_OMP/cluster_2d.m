clear all;
load -ascii points2d_4;
x = points2d_4;
figure(1)
clf
x=x./900;
plot(-x(:,2),x(:,1),'bo');
axis('off');
hold on;
[N, junk] = size(x);
% Number of clusters
K = 200;
mu = mean(x) + 0.01*randn(K,2);
plot(-mu(:,2),mu(:,1),'rx','markersize',16);
hold off;
axis('equal')
% Sigma initialization
sigma = 1.0;
% Number of sigma estimations
number_sigma = 35;
negLogLike = zeros(1,number_sigma);     %initialize negative log liklihood 
kmeanNegLogLike = zeros(1,number_sigma);
for s = 1:number_sigma,
%   Number of iterations for r and mu estimation
  number_iterations = 20;
  for i = 1:number_iterations,
    mup =mu';
    r = exp(-((x(:,1)*ones(1,K)-ones(N,1)*mup(1,:)).^2+(x(:,2)*ones(1,K)-ones(N,1)*mup(2,:)).^2)/(2*sigma^2));
    r = r./((sum(r'))'*ones(1,K));
    mu = r'*x./(sum(r))';
  end
  for m = 1:N/50
        for n = 1:K/10
          p = 1/sigma^junk * exp(-norm(x(m,:)-mu(n,:))^2/(2*sigma^2));
          temp = r(m,n)*log(1*p);
          temp1 = r(m,n)*log(r(m,n));
          negLogLike(1,s) = negLogLike(1,s) - temp + temp1;
        end
  end
  plot(-x(:,2),x(:,1),'bo');
  axis('off');
  hold on;
  plot(-mu(:,2),mu(:,1),'rx','markersize',16);
  title("Mixture Model Clustering");
  axis('off');
  pause(0.1);
  hold off;
  mup = mu';
  xminusmusq = (x(:,1)*ones(1,K)-ones(N,1)*mup(1,:)).^2+(x(:,2)*ones(1,K)-ones(N,1)*mup(2,:)).^2;
  sigma = sqrt(sum(sum(r.*xminusmusq))/(2*N))
  if sigma <= 1.0e-04,
    sigma = 1.0e-03;
  end
  mu = mu + 0.01*randn(K,2);
end
figure(3)
plot(-mu(:,2),mu(:,1),'rx','markersize',16);
title("Mixture Model Clustering");
axis('off');
% K-Means CLustering
mu2 = mean(x) + 0.1*randn(K,2);
number_kmeans = 200;
for i=1:number_kmeans,
  mup = mu2';
  distsq = (x(:,1)*ones(1,K)-ones(N,1)*mup(1,:)).^2+(x(:,2)*ones(1,K)-ones(N,1)*mup(2,:)).^2;
  [y,k] = min(distsq');
  j = 1:N;
  z = zeros(N,K);
  z(j+N*(k-1))=1;
  mu2 = z'*x./(sum(z))';
  figure(2)
  plot(-x(:,2),x(:,1),'bo');
  axis('off');
  hold on;
  plot(-mu2(:,2),mu2(:,1),'rx','markersize',16);
  title("K-Means Clustering");
  axis('off');
  pause(0.1);
  hold off;
  for m = 1:N/50
        for n = 1:K/10
          p = 1/sigma^junk * exp(-norm(x(m,:)-mu(n,:))^2/(2*sigma^2));
          temp = r(m,n)*log(1*p);
          temp1 = r(m,n)*log(r(m,n));
          kmeanNegLogLike(1,s) = kmeanNegLogLike(1,s) - temp + temp1;
        end
  end
end
figure(4)
plot(-mu2(:,2),mu2(:,1),'rx','markersize',16);
title("K-Means Clustering");
axis('off');