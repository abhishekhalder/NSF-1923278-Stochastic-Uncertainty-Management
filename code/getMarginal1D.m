function [X1,marg] = getMarginal1D(X,PDF,nBins)

Grid = linspace(min(X),max(X),nBins+1);

dx = mean(diff(Grid));

 

for i=1:nBins

    ii = find(X>=Grid(i) & X < Grid(i+1));

    marg.MC(i,1) = length(ii); % MC marginal via frequentist counting 

    marg.PF1(i,1) = sum(PDF(ii)); % PF marginal via arithmatic mean

    marg.PF2(i,1) = median(PDF(ii)); % PF marginal via median

    X1(i) = (Grid(i)+Grid(i+1))/2.0;

end

marg.MC = marg.MC/sum(marg.MC)/dx;

marg.PF1 = marg.PF1/sum(marg.PF1)/dx;

marg.PF2 = marg.PF2/sum(marg.PF2)/dx;