close all
Dataset=importdata('nonlinear50');

Y=Dataset(:,1);
X=Dataset(:,2:end);
[l n]=size(X);
eps=1e-6;

C=200;
K=createKmatrix(X,Y);
L=-ones(1,l);
Aeq=Y';
beq=0;
lb=zeros(l,1);
ub=C*ones(l,1);

[lambda,fval]=quadprog(K,L,[],[],Aeq,beq,lb,ub);

wstar=zeros(6,1);  
for i=1:l
  wstar=wstar+(Y(i)*lambda(i))*[1;X(i,1)^2;X(i,2)^2;sqrt(2)*(X(i,1))...
 ;sqrt(2)*(X(i,2));sqrt(2)*(X(i,1)*X(i,2))];
end
idx=find(lambda>eps & lambda<C-eps);
ibar=idx(1);
bstar=1/Y(ibar);
for jj=1:l
 bstar=bstar-(Y(jj)*lambda(jj))*kernel(X(jj,:),X(ibar,:));    
end
% plot separating surface
 fimplicit(@(x1,x2)(wstar(1)*1)+(wstar(2)*(x1^2))+...
(wstar(3)*(x2^2))+(wstar(4)*sqrt(2)*x1)+(wstar(5)*sqrt(2)*x2)+(wstar(6)*...
sqrt(2)*x1*x2)+bstar);
%
hold on;
misclassified=0;
% margin error 0<=xi<=1, misclassification xi>1
for i=1:l
  output=(wstar(1)*1)+(wstar(2)*(X(i,1)^2))+(wstar(3)*(X(i,2)^2))+...
(wstar(4)*sqrt(2)*X(i,1))+(wstar(5)*sqrt(2)*X(i,2))+(wstar(6)*sqrt(2)*...
X(i,1)*X(i,2))+bstar;
 if Y(i)>0
  if output*Y(i)<0
    plot(X(i,1),X(i,2),'r*');
    misclassified=misclassified+1;
  else
   plot(X(i,1),X(i,2),'m*');
  end
 else
  if output*Y(i)<0
    plot(X(i,1),X(i,2),'ro');
    misclassified=misclassified+1;
  else
   plot(X(i,1),X(i,2),'bo');
  end
 end
end
1/(norm(wstar)^2)
misclassified
