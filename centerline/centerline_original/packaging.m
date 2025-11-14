clear all
clc
clear all
% inputPath = uigetdir('*.*','请选择文件夹');
inputPath = '.\A001998940CTA\CTADICOM';
outputPath = '.\TEST';
if ~exist(inputPath)
    disp('没有这个文件夹');
    return 
end
Startfile = '1.2.840.113619.2.327.3.2831165673.377.1498691926.76.127.dcm';
StManPoint = [299,289];
% 以矩阵的坐标系来表示，不是绘图的xy坐标系表示

Endfile = '1.2.840.113619.2.327.3.2831165673.377.1498691926.76.370.dcm';
EndManPoint = [234,328];
% 以矩阵的坐标系来表示，不是绘图的xy坐标系表示

StManPoint=round(StManPoint);
EndManPoint=round(EndManPoint);
Is_show_path_in_binary_image = 0;
Is_show_path_in_gray_image = 0;
Is_show_thresholding = 0;
%% 第一部分读取数据
close all
clc
tic
if inputPath~=0
    pathstr=strcat(inputPath,'\');
    files=dir([inputPath,'\*.dcm']);% 读取文件夹中的文件
    file_num = length(files); %读取文件长度
else
    return;
end

count=0;
h = waitbar(0,'Loading images '); % waiting bar
image_all=[];
clear FileName store_info
for i = 1:file_num
    image= dicomread(strcat(pathstr,files(i).name));%dicomread(strcat(pathstr,files(index(i)+2).name));
    metadata = dicominfo(strcat(pathstr,files(i).name));%存储信息
    %% 首先二值化
    InstanceNumber=metadata.InstanceNumber;
    % ImOri(:,:,InstanceNumber)=image;
    % image(image<=GrayThre1)=0;
    image_all(:,:,InstanceNumber)=image;
    FileName{InstanceNumber}=files(i).name;
    store_info{InstanceNumber}=metadata;
    count=count+1;
    waitbar(count/file_num)
    pct = round(100*count/file_num);
    set(h,'Name',strcat(num2str(pct),'%'));
    if strcmp(files(i).name,Startfile)
       StartInsNum =  InstanceNumber;
    end
    if strcmp(files(i).name,Endfile)
       EndInsNum =  InstanceNumber;
    end
end
close(h)

Zreso=abs(store_info{1}.SliceLocation-store_info{2}.SliceLocation);
Xreso=store_info{1}.PixelSpacing(1);
Yreso=store_info{1}.PixelSpacing(2);
%
Resolu=[Xreso,Yreso,Zreso];
RecaleIntercept=store_info{1}.RescaleIntercept;

if store_info{1}.SliceLocation > store_info{end}.SliceLocation
    Isflip=1;
else
    Isflip=0;
end
if Isflip
    image_all=flipdim(image_all,3);
    EndInsNum = size(image_all,3) - EndInsNum + 1;
    StartInsNum = size(image_all,3) - StartInsNum + 1;
end
if StartInsNum>EndInsNum
    % 如果开始层对应的序号StartInsNum大于结束层对应的序列，则互换一下，不然后面截取会出错
    TempInsNum = StartInsNum;
    TpManPoint = StManPoint;
    StManPoint = EndManPoint;
    StartInsNum = EndInsNum;
    EndInsNum = TempInsNum;
    EndManPoint = TpManPoint;
end
disp(['读取图片:'])
toc

% 
clearvars -except Resolu StartInsNum EndInsNum image_all StManPoint ...
    EndManPoint RecaleIntercept Xreso Zreso outputPath Is_show_path_in_binary_image Is_show_thresholding Is_show_path_in_gray_image
%% 第二部分，计算管径曲线
% global D RouteMat pathOut TopPoint MidPoint Zreso Xreso Yreso StMask EndMask inputPath IMG_intrested 
% global Radius
% 计时第一段
tic
EndSlice = EndInsNum;
StaSlice = StartInsNum;

Point1=[StManPoint,StaSlice];
Point2=[EndManPoint,EndSlice];
showflag = Is_show_thresholding;
[ThreMax1,ThreMin1]=AutomaticThresholding1(image_all,Point1,showflag);
[ThreMax2,ThreMin2]=AutomaticThresholding1(image_all,Point2,showflag);
ThreMax=max(ThreMax1,ThreMax2);
ThreMin=min(ThreMin1,ThreMin2);
Thres1 = ThreMin;
Thres2 = ThreMax;
% 建议添加一个矫正的阈值，以防止这两个AutomaticThresholding计算出现了错误
Thres1 = 125 - RecaleIntercept;
Thres2 = 700 - RecaleIntercept;
disp(['计算阈值数值:'])
toc
% 计时第一段共耗时 1.68s 
%
tic
IMG_intrested=double(image_all(:,:,StaSlice:EndSlice));
HighReloSize=[size(IMG_intrested,1),size(IMG_intrested,2),round(Zreso/Xreso*size(IMG_intrested,3))];
IMG_intrested=imresize3(double(IMG_intrested),HighReloSize,'linear');
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
disp(['图像resize:'])
toc
% 计时第二段共耗时 3.58s
tic
ROISIZE=size(J);
IND1=sub2ind(ROISIZE,Point1(1),Point1(2),1);
IND2=sub2ind(ROISIZE,Point2(1),Point2(2),ROISIZE(3));
cc = bwconncomp(J);
LABEL = labelmatrix(cc);
if LABEL(IND1)~=LABEL(IND2)
    disp('起始点和终止点并不在一个连通域，终止分析！');
    return,
end
J = ismember(LABEL, LABEL(IND1));
disp(['提取起始终止点的连通区域:'])
toc
% 计时第三段共耗时 
%
StMask=J(:,:,1);
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
EndMask=J(:,:,end);
EndMask1=EndMask;
[XXX,YYY]=find(EndMask==1);
Radius=5;
for cont=1:length(XXX)
   if norm([XXX(cont),YYY(cont)]-[EndManPoint(1),EndManPoint(2)])>Radius
       EndMask1(XXX(cont),YYY(cont))=0;
   end
end
EndMask=EndMask1;

% D=bwdistsc(~J); %计算时间比matlab自带的函数慢多了，这两个函数的计算结果基本一致
tic
D=bwdist(~J);
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
% 计时第四段耗时可以忽略
mkdir(outputPath);
disp(['3D距离变换:'])
toc
%
tic
showflag=Is_show_path_in_binary_image; %是否保存中间图片
LeftOrRight=-1; % 对于packaging函数的特有的参数值
[RouteMat,pathOut]=ShortestPath_inuse(D_rev,TopPoint,MidPoint,J,outputPath,LeftOrRight,showflag);
% 计时第五段耗时 21.09s
MinXX=round(min(pathOut(:,1))-7);
MaxXX=round(max(pathOut(:,1))+7);
MinYY=round(min(pathOut(:,2))-7);
MaxYY=round(max(pathOut(:,2))+7);
RouteMat1=RouteMat;
RouteMat1=RouteMat1(MinXX:MaxXX,MinYY:MaxYY,:);
disp(['路径检索:'])
toc
%
pathOut_local_region=[];
for cont=1:size(pathOut,1)
    [XX,YY,ZZ]=ind2sub(size(RouteMat1),find(RouteMat1(:)==cont));
    pathOut_local_region = [pathOut_local_region; XX,YY,ZZ];
end
%
Radius=[];
for cont=1:size(pathOut,1)
    Radius(cont)=D(pathOut(cont,1),pathOut(cont,2),pathOut(cont,3));
end

Isfilter=1;
if Isfilter
    % 是否进行滤波
    winds=3;
    w = gausswin(2*winds+1);
    w = w/sum(w);
    Radius_fil = Radius;
    for cont=1+winds:length(Radius)-winds
        Radius_fil(cont)=sum(Radius(cont-winds:cont+winds).*w');
    end
    % Radius=smooth(Radius,5);
end

%
IMGMAT=IMG_intrested(MinXX:MaxXX,MinYY:MaxYY,:);
% 绘制管径曲线与狭窄位置截图
figure,plot(Radius_fil)
AXIS=axis;
axis([AXIS(1),AXIS(2),0,AXIS(4)]);
[MinRadius,MinInd] = min(Radius_fil);
[MaxRadius,MaxInd] = max(Radius_fil);
hold on
plot(MinInd,Radius_fil(MinInd),'ro');
plot(MaxInd,Radius_fil(MaxInd),'bo');
StePos = pathOut(MinInd,:);
figure,imshow(IMG_intrested(:,:,StePos(3)),[1035 1645])
hold on
plot(StePos(2),StePos(1),'r.')
    
save([outputPath,'\MAT_for_3d_render.mat'],'RouteMat1','pathOut','D','Radius','pathOut_local_region','IMGMAT','Thres1','Thres2');
%%  
Is_save_pathimg = Is_show_path_in_gray_image;
if Is_save_pathimg
    for cont=1:size(IMG_intrested,3)
        figure(1)
        imshow(IMG_intrested(MinXX:MaxXX,MinYY:MaxYY,cont),[1035 1645]);
        % imshow(J(MinXX:MaxXX,MinYY:MaxYY,cont));
        
        set(gcf, 'position', get(0,'ScreenSize'));
        hold on
        [XX,YY]=find(RouteMat1(:,:,cont)~=0);
        plot(YY,XX,'r.','MarkerSize',25);
        hold off
        
        saveas(gcf,[outputPath,'\Path',num2str(cont),'.jpg'],'jpg')
        clf
    end
end
%% 第三部分，保存结果图像
% 读取重建图像的数据
% close all
% load MAT_for_3d_render.mat
% 保存二维截面截图
Is_save_pathimg = Is_show_path_in_gray_image;
if Is_save_pathimg
    for cont=1:size(pathOut_local_region,1)
        % figure(1)
        % imshow(IMGMAT(:,:,pathOut_local_region(cont,3)),[1035 1645]);
        % imshow(IMGMAT(:,:,pathOut_local_region(cont,3)),[1100 1645]);
        imshow(IMGMAT(:,:,pathOut_local_region(cont,3)),[Thres1 Thres2]);
        set(gca,'position',[0 0 1 1]);
        set(gcf, 'position', [1 1 1100 1000]);
        XX=pathOut_local_region(cont,1);
        YY=pathOut_local_region(cont,2);
        hold on,plot(YY,XX,'r.','MarkerSize',25);
        %
        r=Radius(cont);
        theta=0:2*pi/3600:2*pi;
        Circle1=YY+r*cos(theta);
        Circle2=XX+r*sin(theta);
        hold on,plot(Circle1,Circle2,'r','Linewidth',1);
        % 显示外轮廓
        %     r=Radius(cont)+1;
        %     theta=0:2*pi/3600:2*pi;
        %     Circle1=YY+r*cos(theta);
        %     Circle2=XX+r*sin(theta);
        %     hold on,plot(Circle1,Circle2,'m','Linewidth',1);
        
        saveas(gcf,[outputPath,'\Path_and_Circle',num2str(cont),'.png']);
        close all
    end
end

% %% 
% % 保存背景3D重建图片
% % 首先，我们需要将血管附近的像素点保留，其他的不保留。
% CircleR = 10;
% Neib=[];
% for cont=-CircleR:CircleR
%     for cont1=-CircleR:CircleR
%         for cont2=-CircleR:CircleR
%             if (cont^2+cont1^2+cont2^2)<=(CircleR^2)
%                 Neib=[Neib;cont,cont1,cont2];
%             end
%         end
%     end
% end
% VesNeib=zeros(size(pathOut,1)*size(Neib,1),3);
% for cont=1:size(pathOut,1)
%     CurVesNeb(:,1)=pathOut(cont,1)+Neib(:,1);
%     CurVesNeb(:,2)=pathOut(cont,2)+Neib(:,2);
%     CurVesNeb(:,3)=pathOut(cont,3)+Neib(:,3);
%     VesNeib(size(Neib,1)*(cont-1)+1:size(Neib,1)*cont,:)=CurVesNeb;
% end
% DeleInd = find( (VesNeib(:,1)<1)|(VesNeib(:,1)>size(IMG_intrested,1))|(VesNeib(:,2)<1)|(VesNeib(:,2)>size(IMG_intrested,2))...
%     |(VesNeib(:,3)<1)|(VesNeib(:,3)>size(IMG_intrested,3)) );
% VesNeib(DeleInd,:)=[];
% %
% ShowIMG = zeros(size(IMG_intrested));
% VesNeibInd = sub2ind(size(IMG_intrested),VesNeib(:,1),VesNeib(:,2),VesNeib(:,3));
% ShowIMG(VesNeibInd)=IMG_intrested(VesNeibInd);
% ShowIMG = ShowIMG(MinXX:MaxXX,MinYY:MaxYY,:);
% 
% % 该函数在使用时得注意观察的角度要人工事先确认，通过查看以下的VS变量
% % CameraPosition1 = [0.8521 -0.9634 -0.2472];
% % CameraUpVector1 = [0.2754 -0.3077 -0.9108];
% % 循环显示3D重建图像
% DIS = 4;
% CameraPosition_can{1} = [-DIS -DIS 0.5];
% CameraUpVector_can{1} = [1 1 0];
% CameraPosition_can{2} = [-DIS DIS 0.5];
% CameraUpVector_can{2} = [1 -1 0];
% CameraPosition_can{3} = [DIS DIS 0.5];
% CameraUpVector_can{3} = [-1 -1 0];
% CameraPosition_can{4} = [DIS -DIS 0.5];
% CameraUpVector_can{4} = [-1 1 0];
% 
% for cont=1:length(CameraPosition_can)
%     CameraPosition = CameraPosition_can{cont};
%     CameraUpVector = CameraUpVector_can{cont};
%     BonAlpha=1;
%     VslAlpha=0.03;
%     RRatio=3; %放大的倍数
%     Volume=imresize3(IMGMAT,RRatio*size(IMGMAT),'linear');
%     MaxLightness = max(Volume(:));
%     Volume(find(max(Volume(:))))= 2*MaxLightness;
%     intensity = [0 Thres1-1 Thres1 Thres2  MaxLightness 1.5*MaxLightness 2*MaxLightness];
%     BckAlpha=0;
%     alpha = [0 BckAlpha VslAlpha VslAlpha BonAlpha 1 1];
%     color = ([0 0 0; 0 0 0; 255 255 255; 255 255 255; 255 255 255; 255 255 0; 255 255 0]) ./ 255;
%     queryPoints = linspace(min(intensity),max(intensity),256);
%     alphamap = interp1(intensity,alpha,queryPoints)';
%     colormap = interp1(intensity,color,queryPoints);
%     
%     figure(50),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%     saveas(gcf,[outputPath,'\3Dview0_',num2str(cont),'.png']);
% end
% 
% 
% 
%         
% RRatio=3; %放大的倍数
% ALPHA=1:-0.1:0.1;
% ALPHA=[ALPHA,0.06,0.03];
% CameraPosition1 = [0.8521 -0.9634 -0.2472];
% CameraUpVector1 = [0.2754 -0.3077 -0.9108];
% 
% CameraPosition2 = [-0.9813 0.7810 -0.3775];
% CameraUpVector2 = [-0.2998 0.1351 -0.9444];
% istwoviweangle = 1;
% 
% for cont=1:length(ALPHA)
%     VslAlpha=ALPHA(cont);
%     % BonAlpha= (1-0.25)/(1-0.03)*(ALPHA(cont)-0.03)+0.25;
%     BonAlpha=1;
%     Volume=imresize3(IMGMAT,RRatio*size(IMGMAT),'linear');
%     MaxLightness = max(Volume(:));
%     Volume(find(max(Volume(:))))= 2*MaxLightness;
%     intensity = [0 Thres1-1 Thres1 Thres2  MaxLightness 1.5*MaxLightness 2*MaxLightness];
%     % 第一次 1 最后一次0
%     BckAlpha=0.04*(1-   (cont-1)/(length(ALPHA)-1)   );
%     alpha = [0 BckAlpha VslAlpha VslAlpha BonAlpha 1 1];
%     color = ([0 0 0; 0 0 0; 255 255 255; 255 255 255; 255 255 255; 255 255 0; 255 255 0]) ./ 255;
%     queryPoints = linspace(min(intensity),max(intensity),256);
%     alphamap = interp1(intensity,alpha,queryPoints)';
%     colormap = interp1(intensity,color,queryPoints);
%     %
%     if istwoviweangle
%         CameraPosition = CameraPosition1;
%         CameraUpVector = CameraUpVector1;
%         figure(50),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['temp1.png']);
%         %
%         CameraPosition = CameraPosition2;
%         CameraUpVector = CameraUpVector2;
%         figure(51),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['temp2.png']);
%         close all
%         temp1=imread('temp1.png');
%         temp2=imread('temp2.png');
%         Width=size(temp1,2);
%         temp1=temp1(:,round(Width/4):round(Width/4*3),:);
%         temp2=temp2(:,round(Width/4):round(Width/4*3),:);
%         mergedimg = [temp1,temp2];
%         imwrite (mergedimg,['VolShow_back_',num2str(cont),'.png']);
%     else
%         CameraPosition = CameraPosition1;
%         CameraUpVector = CameraUpVector1;
%         figure(50),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['VolShow_back_',num2str(cont),'.png']);
%         close all
%     end
% end
% 
% % 保存球体运动3D重建图片
% RRatio=3; %放大的倍数
% for cont=1:size(pathOut_local_region,1)
%     % 绘制3D图
%     Volume=imresize3(IMGMAT,RRatio*size(IMGMAT),'linear');
%     MaxLightness = max(Volume(:));
%     for cont1=1:cont
%         curPos = RRatio * pathOut_local_region(cont1,:);
%         Neib=findNeib(size(Volume),curPos,ceil(RRatio/2),RRatio/2);
%         for cont3=1:size(Neib,1)
%             Volume(Neib(cont3,1),Neib(cont3,2),Neib(cont3,3))=2*MaxLightness;
%         end
%     end
%     CurrentR= RRatio * Radius(cont);
%     
%     Neib=findNeib(size(Volume),RRatio * pathOut_local_region(cont,:),ceil(CurrentR),CurrentR);
%     for cont2=1:size(Neib,1)
%         Volume(Neib(cont2,1),Neib(cont2,2),Neib(cont2,3))=2*MaxLightness;
%     end
%     %
%     intensity = [0 Thres1-1 Thres1 Thres2  MaxLightness 1.5*MaxLightness 2*MaxLightness];
%     alpha = [0 0 0.03 0.03 1 1 1];
%     color = ([0 0 0; 0 0 0; 255 255 255; 255 255 255; 255 255 255; 255 255 0; 255 255 0]) ./ 255;
%     queryPoints = linspace(min(intensity),max(intensity),256);
%     alphamap = interp1(intensity,alpha,queryPoints)';
%     colormap = interp1(intensity,color,queryPoints);
%     if istwoviweangle
%         CameraPosition = CameraPosition1;
%         CameraUpVector = CameraUpVector1;
%         figure(50),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['temp1.png']);
%         CameraPosition = CameraPosition2;
%         CameraUpVector = CameraUpVector2;
%         figure(51),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['temp2.png']);
%         close all
%         temp1=imread('temp1.png');
%         temp2=imread('temp2.png');
%         Width=size(temp1,2);
%         temp1=temp1(:,round(Width/4):round(Width/4*3),:);
%         temp2=temp2(:,round(Width/4):round(Width/4*3),:);
%         mergedimg = [temp1,temp2];
%         imwrite (mergedimg,['VolShow',num2str(cont),'.png']);
%     else
%         CameraPosition = CameraPosition1;
%         CameraUpVector = CameraUpVector1;
%         figure(50),VS=volshow(Volume,'Colormap',colormap,'Alphamap',alphamap,'CameraPosition',CameraPosition,'CameraUpVector',CameraUpVector);
%         saveas(gcf,['VolShow',num2str(cont),'.png']);
%         close all
%     end
%     
% end
% 
% %保存管径曲线图
% Isfilter=1;
% if Isfilter
%     w = gausswin(7);
%     w = w/sum(w);
%     % Radius = filter(w,1,Radius);
%     Radius = conv(Radius,w,'same');
%     Denominator = conv(ones(size(Radius)),w,'same');
%     Radius = Radius./Denominator;
%     % Radius=smooth(Radius,5);
% end
% 
% for cont=1:length(Radius)
%     figure(15),plot(Radius(1:cont));
%     % set(gca,'border','tight','InitialMagnification','fit');
%     set(gca, 'position', [0.05 0 0.95 1],'ytick',[]);
%     ylabel('Radius')
%     xlim([1 length(Radius)]);
%     ylim([0 1.5*max(Radius)]);
%     hold on
%     plot(cont,Radius(cont),'ro');
%     saveas(gcf,['RadiusCurve',num2str(cont),'.png']);
%     close all
% end










