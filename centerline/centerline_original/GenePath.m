function [RouteMat,pathOut,D,IMG_intrested,TopPoint,MidPoint]=GenePath(image_all,inputPath,StaSlice,EndSlice,Thres1,Thres2,Zreso,Xreso,LeftOrRight,StMask,EndMask)
IMG_intrested=double(image_all(:,:,StaSlice:EndSlice));
HighReloSize=[size(IMG_intrested,1),size(IMG_intrested,2),round(Zreso/Xreso*size(IMG_intrested,3))];
IMG_intrested=imresize3(double(IMG_intrested),HighReloSize,'linear');
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
%
D=bwdistsc(~J);
D(J==0)=0;
MaxVal=max(D(:));
MinVal=min(D(:));
D_rev=-D+MaxVal+MinVal;
Mat=D(:,:,1).*StMask;
[Xind,Yind]=find(Mat==max(Mat(:)));
TopPoint(1)=Xind(1);   TopPoint(2)=Yind(1);
Mat=D(:,:,end).*EndMask;
[Xind,Yind]=find(Mat==max(Mat(:)));
MidPoint(1)=Xind(1);   MidPoint(2)=Yind(1);
%
if LeftOrRight==0
    mkdir([inputPath,'\Left']);
    delete([inputPath,'\Left']);
    mkdir([inputPath,'\Left']);
elseif LeftOrRight==1
    mkdir([inputPath,'\Right']);
    delete([inputPath,'\Right']);
    mkdir([inputPath,'\Right']);
end
%
[RouteMat,pathOut]=ShortestPath_Auto(D_rev,TopPoint,MidPoint,J,inputPath,LeftOrRight);
%
MinXX=round(min(pathOut(:,1))-30);
MaxXX=round(max(pathOut(:,1))+30);
MinYY=round(min(pathOut(:,2))-30);
MaxYY=round(max(pathOut(:,2))+30);
RouteMat1=RouteMat;
RouteMat1=RouteMat1(MinXX:MaxXX,MinYY:MaxYY,:);

%
for cont=1:size(IMG_intrested,3)
    figure(1)
    imshow(IMG_intrested(MinXX:MaxXX,MinYY:MaxYY,cont),[1035 1645]);
    % imshow(J(MinXX:MaxXX,MinYY:MaxYY,cont));
    
    set(gcf, 'position', get(0,'ScreenSize'));
    hold on
    [XX,YY]=find(RouteMat1(:,:,cont)~=0);
    plot(YY,XX,'r.');
    hold off
    
    if LeftOrRight==0
        saveas(gcf,[inputPath,'\Left\Path',num2str(cont),'.jpg'],'jpg')
    elseif LeftOrRight==1
        saveas(gcf,[inputPath,'\Right\Path',num2str(cont),'.jpg'],'jpg')
    end
    clf
end