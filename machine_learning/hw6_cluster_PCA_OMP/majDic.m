function [ result ] = majDic( x,W,z,epsilon,lambda )
oldobj = 0;
for i=1:10
    u = abs(z);
    for n = 1:1000
        z(:,n) = pinv(W'*W+lambda*diag(1/(2*u(:,n))))*W'*x(:,n);
    end
    sum = 0;
    for j = 1:1000
        for k = 1:350
            if u(k,j) < epsilon
                sum = sum + (z(k,j)^2+u(k,j)^2)/(2*epsilon);
            else
                sum = sum + (z(k,j)^2+u(k,j)^2)/(2*u(k,j));
            end
        end
    end
    E = norm(x-W*z)^2;
    F = lambda*sum;
    obj = E+F;
    if mod(100,i)==0
        disp(["epcho = ",i," obj = ",obj])
    end
    if abs(obj - oldobj)<epsilon
        break;
    end
    oldobj = obj;
end
result = z;
end

