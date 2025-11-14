Path='E:\chengxu\（手工选点前提下）全自动狭窄分析数据';
files=dir(Path);

for cont0=60:length(files)
    filedir=files(cont0).name;
    % [Text,OutFileDirR,Flag]=paoshuju(Path,filedir),
    CurFileDir=[Path,'\',filedir];
    if exist([CurFileDir,'\','LeftStEndPoint.mat'])
        load([CurFileDir,'\','LeftStEndPoint.mat']);
        Thres1=125+1024;
        Thres2=700+1024;
        LeftOrRight=0;
        [StMask,EndMask]=GeneMask(Thres1,Thres2,image_all,StaSlice,EndSlice,StManPoint,EndManPoint);
        %%
        [RouteMat,pathOut,D,IMG_intrested,TopPoint,MidPoint]=GenePath(image_all,CurFileDir,StaSlice,EndSlice,Thres1,Thres2,Zreso,Xreso,LeftOrRight,StMask,EndMask);
        %% 备注Xreso和Zreso并没有从图像中自动读取
        [Radius,Stenosis]=GeneStenosis(pathOut,D);
        %%
        SaveResult(CurFileDir,image_all,D,RouteMat,pathOut,StaSlice,EndSlice,Thres1,Thres2,Stenosis,TopPoint,MidPoint,IMG_intrested,LeftOrRight);
        
    end
    if exist([CurFileDir,'\','RightStEndPoint.mat'])
        load([CurFileDir,'\','RightStEndPoint.mat']);
        Thres1=125+1024;
        Thres2=700+1024;
        LeftOrRight=1;
        [StMask,EndMask]=GeneMask(Thres1,Thres2,image_all,StaSlice,EndSlice,StManPoint,EndManPoint);
        %%
        [RouteMat,pathOut,D,IMG_intrested]=GenePath(image_all,CurFileDir,StaSlice,EndSlice,Thres1,Thres2,Zreso,Xreso,LeftOrRight,StMask,EndMask);
        %% 备注Xreso和Zreso并没有从图像中自动读取
        [Radius,Stenosis]=GeneStenosis(pathOut,D);
        %%
        SaveResult(CurFileDir,image_all,D,RouteMat,pathOut,StaSlice,EndSlice,Thres1,Thres2,Stenosis,TopPoint,MidPoint,IMG_intrested,LeftOrRight);
        
    end
    
end














