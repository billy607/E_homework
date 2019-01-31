function [ result ] = fftConv( x,y )
fftX = fft(x,length(x)+length(y)-1);
fftY = fft(y,length(x)+length(y)-1);

H = fftX .* fftY;
h = ifft(H);

result = h;

end