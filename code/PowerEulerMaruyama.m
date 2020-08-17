function Xupd= PowerEulerMaruyama(h,Xold,driftk,nSample,num_Oscillator)
 
gdw = sqrt(h)*[zeros(nSample,num_Oscillator), randn(nSample,num_Oscillator)]; 

Xupd = Xold + (driftk*h) + gdw;
