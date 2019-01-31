function [ result ] = fftConv_err( x,y )
fftX = fft(x,length(x));
fftY = fft(y,length(y));

H = fftX .* fftY;
h = ifft(H);

result = h;

end