clear all; 

I = imread ('clockwork-angels.jpg');
I = double(I)/255;
image = [I(:,:,1)];
for i =1:1000
patches(:,:,i)=zeros(16);
end
x=randperm(1713-16,1000);
y=randperm(3448-16,1000);
for i=1:1000
patches(:,:,i)=image(x(1,i):x(1,i)+15,y(1,i):y(1,i)+15,1);
end

X = zeros(256,1000);
for i=1:1000
X(:,i)=reshape(patches(:,:,i),256,1);
end
X=X-repmat(mean(X),[size(X,1) 1]);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

param.K=350;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=4; % number of threads

param.iter=100;  % let us see what happens after 100 iterations.

tic
D = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);

fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
ImD=displayPatches(D);
subplot(1,3,2);
imagesc(ImD); colormap('gray');
fprintf('objective function: %f\n',R);

z_SPAMS = D'*X;
