function [err,w]=FLDA(f,l,j)
classA=f(l==0,:);
classB=f(l==1,:);
[rowA,colA]=size(classA);
[rowB,colB]=size(classB);
sumA=zeros(1,colA);
sumB=zeros(1,colB);
for i = 1:rowA
    sumA = sumA + classA(i,:);
end
for i = 1:rowB
    sumB = sumB + classB(i,:);
end  
u1=sumA/rowA;
u2=sumB/rowB;
Sw1=zeros(colA,colA);
Sw2=zeros(colB,colB);
for i = 1:rowA
    Sw1 = Sw1 + (classA(i,:)-u1)'*(classA(i,:)-u1);
end
for i = 1:rowB
    Sw2 = Sw2 + (classB(i,:)-u2)'*(classB(i,:)-u2);
end
Sw=Sw1+Sw2;
w=inv(Sw)*(u1-u2)';
figure(j);
h1 = histogram(classA*w);
hold on
h2 = histogram(classB*w);
hold off
threshold = (min(classA*w)+max(classB*w))/2;
err = zeros(150,1);
for i=1:50
    err(i,1)=classA(i,:)*w - threshold;
end
for j=1:100
    err(j+50,1)=threshold - classB(i,:)*w;
end
X=[threshold threshold];
Y=[0 25];
line(X,Y,'Color','red','LineStyle','--')