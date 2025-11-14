Path='F:\chengxu\（手工选点前提下）全自动狭窄分析数据';
files0=dir(Path);

for cont0=139:length(files0)
    filedir=files0(cont0).name;
    % [Text,OutFileDirR,Flag]=paoshuju(Path,filedir),
    CurFileDir=[Path,'\',filedir,'\CTADICOM'];
    files=dir([CurFileDir,'\*.dcm']);% 读取文件夹中的文件
    file_num = length(files); %读取文件长度

    
    count=0;
    h = waitbar(0,'Loading images '); % waiting bar
    % image_all=[];
    FileName={};
    store_info={};
    for i = 1:file_num
        % image= dicomread(strcat(pathstr,files(i).name));%dicomread(strcat(pathstr,files(index(i)+2).name));
        metadata = dicominfo(strcat(CurFileDir,'\',files(i).name));%存储信息
        %% 首先二值化
        InstanceNumber=metadata.InstanceNumber;
        % ImOri(:,:,InstanceNumber)=image;
        % image(image<=GrayThre1)=0;
        % image_all(:,:,InstanceNumber)=image;
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
    if exist([Path,'\',filedir,'\LeftStEndPoint.mat'])
        save([Path,'\',filedir,'\LeftStEndPoint.mat'],'Xreso','Yreso','Zreso','-append');
    end
    if exist([Path,'\',filedir,'\RightStEndPoint.mat'])
        save([Path,'\',filedir,'\RightStEndPoint.mat'],'Xreso','Yreso','Zreso','-append');
    end
end














