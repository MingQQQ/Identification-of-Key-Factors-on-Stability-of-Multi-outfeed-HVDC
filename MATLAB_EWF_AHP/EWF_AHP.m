clc
R=evalin('base','r1');
[m,n] = size(R);
ex=sum(R,1);
R0=[];
for i = 1:(n-1)
 R0(:,i) = (R(:,i)-min(R(:,i)))/(max(R(:,i))-min(R(:,i)));
end
 R0(:,n)=(max(R(:,n))-R(:,n))/(max(R(:,n))-min(R(:,n)));
R2=zeros(m,n);
for i=1:m
    for j=1:n
    R2(i,j)=R0(i,j)/ex(j);
    end
end

R3=R2.*log(R2);
R4=R3;
R4(find(isnan(R4)==1)) = 0;
ex=sum(R4);
ex1=-1/log(m)*ex;
ex2=(1-ex1)/(n-sum(ex1));
ee=sum([0.3628 0.3254 0.3119].*ex2)
ex3=([0.3628 0.3254 0.3119].*ex2)./ee
for i=1:m
R5(i)=100*sum(R0(i,:).*ex2);
end
R6=R5'
R7=[];
for i=1:m
    Rt(i,:)=100*R0(i,:).*ex3;
R7(i)=100*sum(R0(i,:).*ex3);
end
R8=R7'


