function [ result ] = convolution( x,y )
L = length(x)+length(y)-1;

convSeq = zeros(1,L);

X = [x,zeros(1,L-length(x))];

Y = fliplr(y);
Y = [Y,zeros(1,L-length(y))];
Y = circshift(Y,-(length(y)-1));

for i=1:L
    convSeq(i) = dot(X,Y);
    Y = circshift(Y,1);
end

result = convSeq;

end

