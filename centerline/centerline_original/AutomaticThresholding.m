function [MaxVal,MinVal]=AutomaticThresholding(image_all,Pos,showflag)
% Pos=[252,313,270];
% Pos=[240,203,237];
% Pos=[272,211,407];
Wid=40;
Vec1=squeeze(image_all(Pos(1),Pos(2)-Wid:Pos(2)+Wid,Pos(3)));
Vec2=squeeze(image_all(Pos(1)-Wid:Pos(1)+Wid,Pos(2),Pos(3)));
Vec3=squeeze(image_all(Pos(1),Pos(2),Pos(3)-Wid:Pos(3)+Wid));
Vec4=[];
for cont=-Wid:Wid
    Vec4=[Vec4,squeeze(image_all(Pos(1)+cont,Pos(2)+cont,Pos(3)+cont))];
end
Vec5=[];
for cont=-Wid:Wid
    Vec5=[Vec5,squeeze(image_all(Pos(1)+cont,Pos(2)-cont,Pos(3)+cont))];
end
Vec6=[];
for cont=-Wid:Wid
    Vec6=[Vec6,squeeze(image_all(Pos(1)-cont,Pos(2)+cont,Pos(3)+cont))];
end
Vec7=[];
for cont=-Wid:Wid
    Vec7=[Vec7,squeeze(image_all(Pos(1)-cont,Pos(2)-cont,Pos(3)+cont))];
end

% close all
[Center1,St1,End1]=FindThePeak(Vec1,Wid+1,showflag);
MaxVal1=Vec1(Center1);
MinVal2=min(Vec1(St1),Vec1(End1));

[Center2,St2,End2]=FindThePeak(Vec2,Wid+1,showflag);
MaxVal2=Vec2(Center2);
MinVal2=min(Vec2(St2),Vec2(End2));

[Center3,St3,End3]=FindThePeak(Vec3,Wid+1,showflag);
MaxVal3=Vec3(Center3);
MinVal3=min(Vec1(St3),Vec1(End3));

[Center4,St4,End4]=FindThePeak(Vec4,Wid+1,showflag);
MaxVal4=Vec4(Center4);
MinVal4=min(Vec4(St4),Vec4(End4));

[Center5,St5,End5]=FindThePeak(Vec5,Wid+1,showflag);
MaxVal5=Vec5(Center5);
MinVal5=min(Vec5(St5),Vec5(End5));

[Center6,St6,End6]=FindThePeak(Vec6,Wid+1,showflag);
MaxVal6=Vec6(Center6);
MinVal6=min(Vec6(St6),Vec6(End6));

[Center7,St7,End7]=FindThePeak(Vec7,Wid+1,showflag);
MaxVal7=Vec7(Center7);
MinVal7=min(Vec7(St7),Vec7(End7));

MaxVal=max([MaxVal1,MaxVal2,MaxVal3,MaxVal4,MaxVal5,MaxVal6,MaxVal7]);
MinVal=min([MinVal2,MinVal2,MinVal3,MinVal4,MinVal5,MinVal6,MinVal7]);


%% 








%% 
% for cont=1:size(Map,1)
%     Signal(cont)=sum(Map(cont,:));
% end
% figure,plot(Signal)

% M=imshow(Map)
% set(M,'axis',[1 size(Map,2) 1 size(Map,1)])
% axes('position',[0.1 0.1 0.7 0.7])
% axis([1 size(Map,2) 1 size(Map,1)]);



    

% end
% 
% 
% 
% Dif=Max-Min;
% figure,plot(Min)
% hold on,plot(Max)
% plot(Mean)
% plot(Std)
% plot(Dif)
% legend('min','max','mean','std','dif')






