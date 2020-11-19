function Y = Y_admittance_matrix

YY = full(makeYbus(case14));

idx = [1 2 3 6 8];

for i = 1:length(idx)
   
    for j = 1:length(idx)
        
        Y(i,j) = YY(idx(i),idx(j)); 
        
    end
end


