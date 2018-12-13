function [M]=dist(X,Y,non) 
%构造所有变量的Laplace矩阵
n = size(X,2);
L = repmat(sum(X'.*X',2)',n,1) + repmat(sum(X'.*X',2),1,n) - 2*(X'*X);

y=zeros(n,n);
%构造选择矩阵，表示出ij同class
for i=1:n
    for j=i:n
        if Y(i)==Y(j)
            y(i,j) = 1;
            y(j,i) = 1;
        end
    end
end

%选出相同class的Laplace矩阵
Ls_same=L.*y;
 [~,pos] = sort(Ls_same,2,'descend');
 
%选择前k个距离
i=reshape(repmat([1:n]',1,non)',1,non*n);
j=reshape(pos(:,[1:non])',1,non*n);
v=ones(1,non*n);
Ms_same=sparse(i,j,v,n,n);

%选择不同class的Laplace矩阵
%NaN可能有问题
Ls_diff=L.*(ones(n,n)-y);
Ls_diff(find(Ls_diff==0))=NaN;
[~,pod] = sort(Ls_diff,2);
 
%选择前k个距离
i=reshape(repmat([1:n]',1,non)',1,non*n);
j=reshape(pod(:,[1:non])',1,non*n);
v=-1*ones(1,non*n);
Ms_diff=sparse(i,j,v,n,n);
 
Ms = Ms_same+Ms_diff;

 %生成距离矩阵
 M = sparse(n,n);
 [i,j] = find(Ms~=0);
 for m=1:size(i)
        i_m=i(m);
        j_m=j(m);
        M = M+Ms(i_m,j_m)*sparse([i_m,i_m,j_m,j_m],[i_m,j_m,i_m,j_m],[1,-1,-1,1],n,n);
 end
 

M = full(M);

 