clear all;
param.L = 3;   % number of elements in each linear combination.
param.K = 350; % number of dictionary elements
param.numIteration = 50; % number of iteration to execute the K-SVD algorithm.

param.errorFlag = 0; % decompose signals until a certain error is reached. do not use fix number of coefficients.
%param.errorGoal = sigma;
param.preserveDCAtom = 0;
 
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

data = zeros(256,1000);
for i=1:1000
data(:,i)=reshape(patches(:,:,i),256,1);
end
data=data-repmat(mean(data),[size(data,1) 1]);
data=data ./ repmat(sqrt(sum(data.^2)),[size(data,1) 1]);

%%%%%%%% initial dictionary: Dictionary elements %%%%%%%%
param.InitializationMethod =  'DataElements';

param.displayProgress = 1;
disp('Starting to  train the dictionary');

[Dictionary,output]  = KSVD(data,param);

%%%%%%%%%%%%%%%% Principal Component Analysis %%%%%%%%%%
C=zeros(256);
for i=1:1000
xi=reshape(patches(:,:,i),256,1);
C=C+xi*xi';
end
[eigenvector,eigenvalue]=eig(C);
eiValue=wrev(diag(eigenvalue));
eiVector = fliplr(eigenvector);
Top64=eiVector(:,1:64);
FFt=Top64*Top64';
reImage = image;
err = 0;
for i=1:1000
    err = err + norm(reshape(patches(:,:,i),256,1) - FFt*reshape(patches(:,:,i),256,1));
end
err = err/1000;
disp(['PCA error is ',num2str(err)]);




