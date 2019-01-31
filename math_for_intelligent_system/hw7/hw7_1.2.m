image = imread('hendrix_final.png');
image = double(image);
imageR=[image(:,:,1)];
[m,n]=size(imageR);
Q=zeros(m,n);
R=zeros(m,n);
for j=1:n
v=imageR(:,j);
for i=1:j-1
R(i,j)=Q(:,i)'*imageR(:,j);
v=v-R(i,j)*Q(:,i);
end
R(j,j)=norm(v);
Q(:,j)=v/R(j,j);
end