function output_matrix = SinkhornOMT(input_matrix, rho_x, rho_y)

v = ones(length(rho_x),1);

max_iter = 1000; tol = 1e-5;

iter = 1;

while iter < max_iter
    
    u = rho_y ./ (input_matrix*v);
    
    v = rho_x ./ (input_matrix'*u);
    
    output_matrix = diag(v)*input_matrix*diag(u);
    
    if (norm(output_matrix'*ones(length(rho_x),1) - rho_y,inf) < tol) && (norm(output_matrix*ones(length(rho_y),1) - rho_x,inf) < tol)
        break;
    else
        iter = iter + 1;
    end
    
end

