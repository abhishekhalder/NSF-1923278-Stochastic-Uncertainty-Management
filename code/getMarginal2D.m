function [XX1,XX2,marg] = getMarginal2D(X1,X2,PDF,nBins)

Bins.x1 = linspace(min(X1),max(X1),nBins+1);
Bins.x2 = linspace(min(X2),max(X2),nBins+1);

dx1 = mean(diff(Bins.x1));
dx2 = mean(diff(Bins.x2));

A = dx1*dx2;

[x1,x2] = meshgrid(Bins.x1,Bins.x2);


for i=1:nBins
    for j=1:nBins
        ii = find(X1>= x1(i,j) & X1 < x1(i+1,j+1));
        jj = find(X2(ii)>= x2(i,j) & X2(ii) < x2(i+1,j+1));
        marg.MC(i,j) = length(jj);
        marg.PF(i,j) = sum(PDF(ii(jj)));
        XX1(i,j) = (x1(i,j) +  x1(i+1,j+1))/2.0;
        XX2(i,j) = (x2(i,j) +  x2(i+1,j+1))/2.0;
    end
end

marg.MC = marg.MC/sum(marg.MC(:))/A;
marg.PF = marg.PF/sum(marg.PF(:))/A;