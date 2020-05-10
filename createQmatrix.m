function [Q]=createQmatrix(X,Y)
 [l n]=size(X);
 Q=zeros(l,l);
 for i=1:l
  for j=i:l
    xi=X(i,:);
    xj=X(j,:);
    Q(i,j)=(Y(i)*Y(j))*(xi*xj');
    Q(j,i)=Q(i,j);
   end
  end

end 


