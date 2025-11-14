function [Radius,Stenosis]=GeneStenosis(pathOut,D)
for cont=1:size(pathOut,1)
    Radius(cont)=D(pathOut(cont,1),pathOut(cont,2),pathOut(cont,3));
end

for cont=length(Radius):-1:1
    RadPerc(cont)=Radius(cont)/max(Radius(1:cont));
end
StenoPer=1-RadPerc;
[Stenosis,index]=max(StenoPer);

figure(3),plot(Radius);
hold on,
plot(index(1),Radius(index(1)),'r*')
hold off
disp(['œ¡’≠≥Ã∂»Œ™',num2str(Stenosis)]);