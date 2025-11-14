function SaveResult(inputPath,image_all,D,RouteMat,pathOut,StaSlice,EndSlice,Thres1,Thres2,Stenosis,TopPoint,MidPoint,IMG_intrested,LeftOrRight)



if LeftOrRight==0
    
    figure(3),
    saveas(gcf,[inputPath,'\Left\','Left_Radius','.jpg'],'jpg')
    
    fid = fopen([inputPath,'\Left\','Left_Stenosis.txt'],'w');
    fprintf(fid,'%f\n',Stenosis);
    fclose(fid);
    
    save([inputPath,'\Left\Left_Result.mat'],'image_all','IMG_intrested','EndSlice','StaSlice','Thres1','Thres2','D','pathOut','RouteMat','Stenosis','TopPoint','MidPoint')
    
elseif LeftOrRight==1
    
    figure(3),
    saveas(gcf,[inputPath,'\Right\','Right_Radius','.jpg'],'jpg')
    
    fid = fopen([inputPath,'\Right\','Right_Stenosis.txt'],'w');
    fprintf(fid,'%f\n',Stenosis);
    fclose(fid);
    
    save([inputPath,'\Right\Right_Result.mat'],'image_all','IMG_intrested','EndSlice','StaSlice','Thres1','Thres2','D','pathOut','RouteMat','Stenosis','TopPoint','MidPoint')
    
else
    
end