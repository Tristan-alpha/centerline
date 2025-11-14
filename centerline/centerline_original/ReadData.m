inputPath='G:\chengxu\狭窄自动分析2018.7.26\Code\201801080936A002165109CTA\CTADICOM';
if inputPath~=0
    pathstr=strcat(inputPath,'\');
    files=dir(inputPath);% 读取文件夹中的文件
    file_num = length(files)-2; %读取文件长度
else
    return;
end

tic
count=0;
h = waitbar(0,'Loading images '); % waiting bar
for i = 1:file_num
    image= dicomread(strcat(pathstr,files(i+2).name));%dicomread(strcat(pathstr,files(index(i)+2).name));
    metadata = dicominfo(strcat(pathstr,files(i+2).name));%存储信息
    %% 首先二值化
    InstanceNumber=metadata.InstanceNumber;
    % ImOri(:,:,InstanceNumber)=image;
    % image(image<=GrayThre1)=0;
    image_all(:,:,InstanceNumber)=image;
    FileName{InstanceNumber}=files(i+2).name;
    store_info{InstanceNumber}=metadata;
    count=count+1;
    waitbar(count/file_num)
    pct = round(100*count/file_num);
    set(h,'Name',strcat(num2str(pct),'%'));
end


Zreso=abs(store_info{1}.SliceLocation-store_info{2}.SliceLocation);
Xreso=store_info{1}.PixelSpacing(1);
Yreso=store_info{1}.PixelSpacing(2);

Resolu=[Xreso,Yreso,Zreso];

% 

RecaleIntercept=store_info{1}.RescaleIntercept;

close(h)


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
    image_all=flipdim(image_all,3);
    for cont=1:size(image_all,3)
        image_temp(:,:,cont)=image_all(:,end:-1:1,cont);
    end
    image_all=image_temp;
end

clear image_temp

