function K = coupling_matrix(Y,E)

m = size(Y);
nc = m(2);

for i = 1:nc
       
  for j = 1:nc
   
     if i == j
         
         K(i,j) = 0;
         
     else 
         K(i,j) = E(i)*E(j)*abs(Y(i,j));
     end
  end
    
    
end