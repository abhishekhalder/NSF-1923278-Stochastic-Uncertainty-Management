function Xupd= PowerEulerMaruyama(h,Xold,driftk,nSample,dim)
 

gdw = sqrt(h)*[zeros(nSample,dim/2),randn(nSample,dim/2)]; 


%g=[zeros(dim/2),zeros(dim/2);zeros(dim/2),eye(dim/2)];

    %for i=1:nSample

       % Xupd(i,:) = Xold(i,:) + (driftk*h) +g*dw(i,:)';

    %end
    Xupd = Xold+ (driftk*h) +gdw;
end
