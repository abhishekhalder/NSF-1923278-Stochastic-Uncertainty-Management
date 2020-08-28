function Xupd= PowerEulerMaruyama(h,Xold,driftk,nSample,num_Oscillator)
 
gdw = sqrt(h)*[zeros(nSample,num_Oscillator), randn(nSample,num_Oscillator)]; 

Xupd = Xold + (driftk*h) + gdw;

Xupd = [wrapTo2Pi(Xupd(:,1:num_Oscillator)),Xupd(:,num_Oscillator+1:2*num_Oscillator)]; 
