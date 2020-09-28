function phi = phase_shift_matrix(Y)

m = size(Y);
nc = m(2);

for i = 1:nc
    
   
  for j = 1:nc
   
     if i == j
         
         phi(i,j) = 0;
         
     else 
         phi(i,j) = -atan2(real(Y(i,j)),imag(Y(i,j)));
     end
  end
    
    
end