function [StMask,EndMask]=GeneMask(Thres1,Thres2,image_all,StaSlice,EndSlice,StManPoint,EndManPoint)
Thres1=125+1024;
Thres2=700+1024;
% ¶ÁÈ¡Î»ÖÃ
IMG_intrested=double(image_all(:,:,StaSlice));
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
Label=bwlabel(J);
Val=Label(StManPoint(1),StManPoint(2));
StMask=(Label==Val);
StMask1=StMask;
[XXX,YYY]=find(StMask==1);
Radius=5;
for cont=1:length(XXX)
    if norm([XXX(cont),YYY(cont)]-[StManPoint(1),StManPoint(2)])>Radius
        StMask1(XXX(cont),YYY(cont))=0;
    end
end
StMask=StMask1;
%
IMG_intrested=double(image_all(:,:,EndSlice));
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
Label=bwlabel(J);
Val=Label(EndManPoint(1),EndManPoint(2));
EndMask=(Label==Val);
EndMask1=EndMask;
[XXX,YYY]=find(EndMask==1);
Radius=5;
for cont=1:length(XXX)
    if norm([XXX(cont),YYY(cont)]-[EndManPoint(1),EndManPoint(2)])>Radius
        EndMask1(XXX(cont),YYY(cont))=0;
    end
end
EndMask=EndMask1;