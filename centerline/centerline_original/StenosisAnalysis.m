function varargout = StenosisAnalysis(varargin)
% STENOSISANALYSIS MATLAB code for StenosisAnalysis.fig
%      STENOSISANALYSIS, by itself, creates a new STENOSISANALYSIS or raises the existing
%      singleton*.
%
%      H = STENOSISANALYSIS returns the handle to a new STENOSISANALYSIS or the handle to
%      the existing singleton*.
%
%      STENOSISANALYSIS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STENOSISANALYSIS.M with the given input arguments.
%
%      STENOSISANALYSIS('Property','Value',...) creates a new STENOSISANALYSIS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before StenosisAnalysis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to StenosisAnalysis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help StenosisAnalysis

% Last Modified by GUIDE v2.5 19-Aug-2018 19:00:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @StenosisAnalysis_OpeningFcn, ...
    'gui_OutputFcn',  @StenosisAnalysis_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before StenosisAnalysis is made visible.
function StenosisAnalysis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to StenosisAnalysis (see VARARGIN)

% Choose default command line output for StenosisAnalysis
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes StenosisAnalysis wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = StenosisAnalysis_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global image_all inputPath Xreso Yreso Zreso

inputPath = uigetdir('*.*','请选择文件夹');
% inputPath='G:\chengxu\狭窄自动分析2018.7.26\Code\201801080936A002165109CTA\CTADICOM';
if inputPath~=0
    pathstr=strcat(inputPath,'\');
    files=dir([inputPath,'\*.dcm']);% 读取文件夹中的文件
    file_num = length(files); %读取文件长度
else
    return;
end

tic
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
end
close(h)

Zreso=abs(store_info{1}.SliceLocation-store_info{2}.SliceLocation);
Xreso=store_info{1}.PixelSpacing(1);
Yreso=store_info{1}.PixelSpacing(2);
%
Resolu=[Xreso,Yreso,Zreso];
RecaleIntercept=store_info{1}.RescaleIntercept;

StPixel=sum(sum(imfill(image_all(:,:,1)>500,'holes')));
EndPixel=sum(sum(imfill(image_all(:,:,end)>500,'holes')));

% for cont=1:size(image_all,3)
%     SK(cont)=sum(sum(image_all(:,:,cont)));
% end

% if sum(SK(1:round(length(SK)/2)))<sum(SK(round(length(SK)/2):end))
if StPixel<EndPixel
    Isflip=0;
else
    Isflip=1;
end

if Isflip
    image_all=image_all(:,:,end:-1:1);
%     image_all=flipdim(image_all,3);
%     for cont=1:size(image_all,3)
%         image_temp(:,:,cont)=image_all(:,end:-1:1,cont);
%     end
%     image_all=image_temp;
end
toc
% clear image_temp
% set(handles.slider1,'Max',file_num);
% set(handles.slider1,'Min',1);
set(handles.slider1,'Max',file_num,'Min',1,'Value',1)
set(handles.text7,'String',num2str(1));
axes(handles.axes1)
imshow(image_all(:,:,1),[1035 1645]);








% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image_all inputPath Xreso Yreso Zreso
figure,XimShow_Vessel(double(image_all),[1035 1645]);




function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global StaSlice image_all StMask Thres1 Thres2 StManPoint
str = get(handles.edit1,'string');
StaSlice = str2num(str);
str = get(handles.edit3,'string');
Thres1 = str2num(str)+1024;
str = get(handles.edit4,'string');
Thres2 = str2num(str)+1024;

IMG_intrested=double(image_all(:,:,StaSlice));
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
Label=bwlabel(J);
figure(1),imshow(IMG_intrested,[]);
figure(2),imshow(J,[]);
[Xpos,Ypos]=ginput(1);
Val=Label(round(Ypos),round(Xpos));
StManPoint=[round(Ypos),round(Xpos)];
StMask=(Label==Val);
StMask1=StMask;
[XXX,YYY]=find(StMask==1);
Radius=5;
for cont=1:length(XXX)
   if norm([XXX(cont),YYY(cont)]-[Ypos,Xpos])>Radius
       StMask1(XXX(cont),YYY(cont))=0;
   end
end
StMask=StMask1;




% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global EndSlice image_all EndMask Thres1 Thres2 EndManPoint
str = get(handles.edit2,'string');
EndSlice = str2num(str);
str = get(handles.edit3,'string');
Thres1 = str2num(str)+1024;
str = get(handles.edit4,'string');
Thres2 = str2num(str)+1024;

IMG_intrested=double(image_all(:,:,EndSlice));
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
Label=bwlabel(J);
figure(1),imshow(IMG_intrested,[]);
figure(2),imshow(J,[]);
[Xpos,Ypos]=ginput(1);
Val=Label(round(Ypos),round(Xpos));
EndManPoint=[round(Ypos),round(Xpos)];
EndMask=(Label==Val);
EndMask1=EndMask;
[XXX,YYY]=find(EndMask==1);
Radius=5;
for cont=1:length(XXX)
   if norm([XXX(cont),YYY(cont)]-[Ypos,Xpos])>Radius
       EndMask1(XXX(cont),YYY(cont))=0;
   end
end
EndMask=EndMask1;

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global inputPath image_all D RouteMat pathOut EndSlice StaSlice Thres1 Thres2 Stenosis TopPoint MidPoint IMG_intrested

str = get(handles.edit5,'string');
LeftOrRight = str2num(str);

if LeftOrRight==0
    
    figure(3),
    saveas(gcf,[inputPath,'\..\Left\','Left_Radius','.jpg'],'jpg')
    
    fid = fopen([inputPath,'\..\Left\','Left_Stenosis.txt'],'w');
    fprintf(fid,'%f\n',Stenosis);
    fclose(fid);
    
    save([inputPath,'\..\Left\Left_Result.mat'],'image_all','IMG_intrested','EndSlice','StaSlice','Thres1','Thres2','D','pathOut','RouteMat','Stenosis','TopPoint','MidPoint')
    
elseif LeftOrRight==1
    
    figure(3),
    saveas(gcf,[inputPath,'\..\Right\','Right_Radius','.jpg'],'jpg')
    
    fid = fopen([inputPath,'\..\Right\','Right_Stenosis.txt'],'w');
    fprintf(fid,'%f\n',Stenosis);
    fclose(fid);
    
    save([inputPath,'\..\Right\Right_Result.mat'],'image_all','IMG_intrested','EndSlice','StaSlice','Thres1','Thres2','D','pathOut','RouteMat','Stenosis','TopPoint','MidPoint')
    
else
    
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image_all EndSlice StaSlice D RouteMat pathOut Thres1 Thres2 TopPoint MidPoint Zreso Xreso Yreso StMask EndMask inputPath IMG_intrested 
global EndManPoint EndSlice StManPoint StaSlice Radius
% 计时第一段
tic
str = get(handles.edit3,'string');
Thres1 = str2num(str)+1024;
str = get(handles.edit4,'string');
Thres2 = str2num(str)+1024;

Point1=[StManPoint,StaSlice];
Point2=[EndManPoint,EndSlice];
showflag = 1;
[ThreMax1,ThreMin1]=AutomaticThresholding1(image_all,Point1,showflag);
[ThreMax2,ThreMin2]=AutomaticThresholding1(image_all,Point2,showflag);
ThreMax=max(ThreMax1,ThreMax2);
ThreMin=min(ThreMin1,ThreMin2);
Thres1 = ThreMin;
Thres2 = ThreMax;
toc
% 计时第一段共耗时 1.68s 
%
tic
IMG_intrested=double(image_all(:,:,StaSlice:EndSlice));
HighReloSize=[size(IMG_intrested,1),size(IMG_intrested,2),round(Zreso/Xreso*size(IMG_intrested,3))];
IMG_intrested=imresize3(double(IMG_intrested),HighReloSize,'linear');
J=(IMG_intrested>Thres1)&(IMG_intrested<Thres2);
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
toc
% 计时第三段共耗时 

% D=bwdistsc(~J); %计算时间比matlab自带的函数慢多了，这两个函数的计算结果基本一致
tic
D=bwdist(~J);
D(J==0)=0;
MaxVal=max(D(:));
MinVal=min(D(:));
D_rev=-D+MaxVal+MinVal;
% toc
% 计时第三段共耗时 5min 32.73s 基本上全部来源于bwdistsc函数
% tic
Mat=D(:,:,1).*StMask;
[Xind,Yind]=find(Mat==max(Mat(:)));
TopPoint(1)=Xind(1);   TopPoint(2)=Yind(1);
Mat=D(:,:,end).*EndMask;
[Xind,Yind]=find(Mat==max(Mat(:)));
MidPoint(1)=Xind(1);   MidPoint(2)=Yind(1);
% 计时第四段耗时可以忽略
str = get(handles.edit5,'string');
LeftOrRight=str2num(str);
if LeftOrRight==0
    mkdir([inputPath,'\..\Left']);
    % delete([inputPath,'\..\Left']);
    % mkdir([inputPath,'\..\Left']);
elseif LeftOrRight==1
    mkdir([inputPath,'\..\Right']);
    % delete([inputPath,'\..\Right']);
    % mkdir([inputPath,'\..\Right']);
end
toc
%
tic
showflag=0; %是否保存中间图片
[RouteMat,pathOut]=ShortestPath_inuse(D_rev,TopPoint,MidPoint,J,inputPath,LeftOrRight,showflag);
% 计时第五段耗时 21.09s
MinXX=round(min(pathOut(:,1))-7);
MaxXX=round(max(pathOut(:,1))+7);
MinYY=round(min(pathOut(:,2))-7);
MaxYY=round(max(pathOut(:,2))+7);
RouteMat1=RouteMat;
RouteMat1=RouteMat1(MinXX:MaxXX,MinYY:MaxYY,:);
toc
%%
pathOut_local_region=[];
for cont=1:size(pathOut,1)
    [XX,YY,ZZ]=ind2sub(size(RouteMat1),find(RouteMat1(:)==cont));
    pathOut_local_region = [pathOut_local_region; XX,YY,ZZ];
end
%%
Radius=[];
for cont=1:size(pathOut,1)
    Radius(cont)=D(pathOut(cont,1),pathOut(cont,2),pathOut(cont,3));
end
Isfilter=0;
if Isfilter
     w = gausswin(6);
     w = w/sum(w);
     Radius = filter(w,1,Radius);
    % Radius=smooth(Radius,5);
end
%%
IMGMAT=IMG_intrested(MinXX:MaxXX,MinYY:MaxYY,:);
if LeftOrRight==0
    save([inputPath,'\..\Left\MAT_for_3d_render.mat'],'RouteMat1','pathOut','D','Radius','pathOut_local_region','IMGMAT','Thres1','Thres2');
elseif LeftOrRight==1
    save([inputPath,'\..\Right\MAT_for_3d_render.mat'],'RouteMat1','pathOut','D','Radius','pathOut_local_region','IMGMAT','Thres1','Thres2')
end

%%

for cont=1:size(IMG_intrested,3)
    figure(1)
    imshow(IMG_intrested(MinXX:MaxXX,MinYY:MaxYY,cont),[1035 1645]);
    % imshow(J(MinXX:MaxXX,MinYY:MaxYY,cont));
    
    set(gcf, 'position', get(0,'ScreenSize'));
    hold on
    [XX,YY]=find(RouteMat1(:,:,cont)~=0);
    plot(YY,XX,'r.','MarkerSize',25);
    hold off
    
    if LeftOrRight==0
        saveas(gcf,[inputPath,'\..\Left\Path',num2str(cont),'.jpg'],'jpg')
    elseif LeftOrRight==1
        saveas(gcf,[inputPath,'\..\Right\Path',num2str(cont),'.jpg'],'jpg')
    end
    clf
end




% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global D pathOut StenoPer Radius Stenosis

for cont=length(Radius):-1:1
    RadPerc(cont)=Radius(cont)/max(Radius(1:cont));
end
StenoPer=1-RadPerc;
[Stenosis,index]=max(StenoPer);

figure(20),plot(Radius(6:end));
hold on,
% plot(index(1),Radius(index(1)),'r*')
disp(['狭窄程度为',num2str(Stenosis)]);






function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global image_all
Slice=get(handles.slider1,'Value'),
set(handles.text7,'String',num2str(round(Slice)));
axes(handles.axes1)
imshow(image_all(:,:,round(Slice)),[1035 1645]);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global inputPath image_all EndSlice StaSlice EndManPoint StManPoint
str = get(handles.edit5,'string');
LeftOrRight=str2num(str);

if LeftOrRight==0
    save([inputPath,'\..\LeftStEndPoint.mat'],'StaSlice','StManPoint','EndManPoint','EndSlice','image_all')
elseif LeftOrRight==1
    save([inputPath,'\..\RightStEndPoint.mat'],'StaSlice','StManPoint','EndManPoint','EndSlice','image_all')
end
