function [blocks,idx] = my_im2col(I,blkSize,slidingDis);
if (slidingDis==1)
    blocks = im2col(I,blkSize,'sliding');
    idx = [1:size(blocks,2)];%inx为1：size(blocks,2)（block的列数）的向量
    return
end

idxMat = zeros(size(I)-blkSize+1);%生成一个m1×m2的矩阵的0矩阵 m1为I的行数减去blksize的第一项再加1 列数同理
idxMat([[1:slidingDis:end-1],end],[[1:slidingDis:end-1],end]) = 1; % 将indmat矩阵的行从1每隔slidingDis直到end-1和最后一个元素赋值为1 同理列也是如此 take blocks in distances of 'slidingDix', but always take the first and last one (in each row and column).
idx = find(idxMat);%找出其非零元的在矩阵中的位置
[rows,cols] = ind2sub(size(idxMat),idx);%将inx的序号对应于idxMat中的序号存放于rows和的cols中
blocks = zeros(prod(blkSize),length(idx));%生成一个blkSize(1)×blkSize(2)为行 inx为列零矩阵
for i = 1:length(idx)
    currBlock = I(rows(i):rows(i)+blkSize(1)-1,cols(i):cols(i)+blkSize(2)-1);%将image中第rows的第i行到第i行+blkSize(1)-1行 cols的第i列到cols+blkSize(2)-1列赋给currBlock 
    blocks(:,i) = currBlock(:);%currBlock转化为向量赋给blocks
end
%已经理解
