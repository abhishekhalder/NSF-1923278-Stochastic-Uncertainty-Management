function Xupd= PowerEulerMaruyama(h,Xold,driftk,nSample,num_Oscillator)
 
global M Sigma
gdw = sqrt(h)*[zeros(nSample,num_Oscillator), randn(nSample,num_Oscillator)]; 

Xupd = Xold + (driftk*h) + gdw;

Xupd(:,1:num_Oscillator) = wrapTo2Pi(Xupd(:,1:num_Oscillator)); 

for ii=1:num_Oscillator

Xupd(:,ii) = wrapTo2PiMSigma(M(ii,ii)/Sigma(ii,ii)*Xupd(:,ii),M(ii,ii),Sigma(ii,ii));

end