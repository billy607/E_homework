clear
clc
data = importdata('dataSet.dat');
data = data';
data = data(1:150,1:4);
label1 = [ones(50,1);zeros(100,1)];
label2 = [zeros(50,1);ones(50,1);zeros(50,1)];
label3 = [zeros(100,1);ones(50,1)];
[err1,w1] = FLDA(data,label1,1);
[err2,w2] = FLDA(data,label2,2);
[err3,w3] = FLDA(data,label3,3);